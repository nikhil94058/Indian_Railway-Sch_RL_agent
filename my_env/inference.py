"""
inference.py  —  Indian Railway Scheduling RL Agent
====================================================
"""

import os
import sys
import json
import heapq
import re
from pathlib import Path
from typing import List, Optional, Dict

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Force UTF-8 on Windows console so → and ≥ in feedback strings don't crash prints
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 1. PROTOCOL CONFIGURATION ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
API_KEY      = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip()

llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
ENV_URL    = os.getenv("ENV_URL", "http://localhost:8000").strip()

# ── 2. PROTOCOL LOGGERS ───────────────────────────────────────────────────
def _format_action(action_dict: dict) -> str:
    return json.dumps(action_dict, separators=(",", ":"), sort_keys=True)

def _format_error(error: Optional[str]) -> str:
    if not error:
        return "null"
    # Replace non-ASCII (→, ≥, etc.) so logs stay clean on any console encoding
    safe = str(error).encode("ascii", errors="replace").decode("ascii")
    return "_".join(safe.split())[:300]

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = _format_error(error)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    safe_score  = max(0.001, min(0.999, score))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={safe_score:.3f} rewards={rewards_str}", flush=True)

# ── 3. PROJECT IMPORTS ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from my_env.client import MyEnv
from my_env.models import TrainAction, StopTime, ScheduleProposal
from my_env.data.network import data as net_data
from my_env.data.config import TRAIN_SPECS, DEFAULT_TRAIN_SPEC, ZONE_WEATHER
from my_env.server.my_env_environment import TASK_REGISTRY, grade_task

# ── 4. NETWORK LOOKUP TABLES (built once at import) ───────────────────────
_TRACK_MAP: Dict[tuple, dict] = {}
for _t in net_data["tracks"]:
    _TRACK_MAP[(_t["from"], _t["to"])] = _t
    _TRACK_MAP[(_t["to"], _t["from"])] = _t

_STATION_ZONE: Dict[str, str] = {s["id"]: s["zone"]           for s in net_data["stations"]}
_STATION_PLAT: Dict[str, int] = {s["id"]: s["platform_count"] for s in net_data["stations"]}

# ── 5. ROUTE CALCULATOR ───────────────────────────────────────────────────
def calculate_route(src: str, dest: str) -> List[str]:
    graph: Dict[str, list] = {}
    for t in net_data["tracks"]:
        graph.setdefault(t["from"], []).append((t["to"],   t["travel_minutes"]))
        graph.setdefault(t["to"],   []).append((t["from"], t["travel_minutes"]))

    queue, visited = [(0, src, [])], set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == dest:
            return path
        for nb, w in graph.get(node, []):
            if nb not in visited:
                heapq.heappush(queue, (cost + w, nb, path))
    return [src, dest]

# ── 6. WEATHER-ADJUSTED SEGMENT TRAVEL TIME ───────────────────────────────
def get_seg_minutes(from_st: str, to_st: str, train_type: str = "Express") -> int:
    """Return the physics-floor travel time for one segment (weather multiplier applied)."""
    track = _TRACK_MAP.get((from_st, to_st))
    if not track:
        return 60
    base      = track["travel_minutes"]
    dist_km   = track.get("distance_km", base)
    zone      = _STATION_ZONE.get(to_st, "NCR")
    mult      = ZONE_WEATHER.get(zone, {}).get("multiplier", 1.0)
    max_spd   = TRAIN_SPECS.get(train_type, DEFAULT_TRAIN_SPEC)["max_speed_kmh"]
    # Stricter of: weather floor (90 % tolerance as physics engine uses) OR speed-physics floor
    weather_floor = int(base * mult * 0.90)
    speed_floor   = int((dist_km / max_spd) * 60)
    return max(weather_floor, speed_floor) + 2   # +2 min safety margin

# ── 7. FALLBACK STOP-TIME BUILDER ─────────────────────────────────────────
def build_fallback_stops(route: List[str], start_time: str, train_type: str) -> List[dict]:
    """
    Calculate valid stop times purely from network physics.
    This is both the prompt baseline AND the emergency fallback when LLM output is invalid.
    """
    spec      = TRAIN_SPECS.get(train_type, DEFAULT_TRAIN_SPEC)
    min_dwell = spec["min_dwell"]

    h, m    = map(int, start_time.split(":"))
    cur_min = h * 60 + m

    stops = []
    for i, station in enumerate(route):
        is_first = (i == 0)
        is_last  = (i == len(route) - 1)

        # Safe platform: cycle within station capacity
        max_plat = _STATION_PLAT.get(station, 4)
        plat     = (i % max_plat) + 1

        if is_first:
            day  = cur_min // 1440
            tod  = cur_min % 1440
            tstr = f"{tod // 60:02d}:{tod % 60:02d}"
            stops.append({"station": station, "arrival": None, "departure": tstr,
                           "platform": plat, "day_offset": day})
            if not is_last:
                cur_min += get_seg_minutes(route[i], route[i + 1], train_type)

        elif is_last:
            day  = cur_min // 1440
            tod  = cur_min % 1440
            tstr = f"{tod // 60:02d}:{tod % 60:02d}"
            stops.append({"station": station, "arrival": tstr, "departure": None,
                           "platform": plat, "day_offset": day})

        else:
            arr_day = cur_min // 1440
            arr_tod = cur_min % 1440
            arr_str = f"{arr_tod // 60:02d}:{arr_tod % 60:02d}"

            dep_min = cur_min + min_dwell
            dep_day = dep_min // 1440
            dep_tod = dep_min % 1440
            dep_str = f"{dep_tod // 60:02d}:{dep_tod % 60:02d}"

            stops.append({"station": station, "arrival": arr_str, "departure": dep_str,
                           "platform": plat, "day_offset": arr_day})
            cur_min = dep_min + get_seg_minutes(route[i], route[i + 1], train_type)

    return stops

# ── 8. LLM PROMPT BUILDER ─────────────────────────────────────────────────
def build_prompt(
    train_def:           dict,
    route:               List[str],
    committed_timetable: List[dict],
    task_def:            dict,
    fallback_stops:      List[dict],
) -> str:
    train_id   = train_def["id"]
    train_type = train_def["type"]
    departure  = train_def["time"]

    spec      = TRAIN_SPECS.get(train_type, DEFAULT_TRAIN_SPEC)
    min_dwell = spec["min_dwell"]
    max_dwell = spec["max_dwell"]
    min_hw    = spec["min_headway"]

    # Segment travel times (weather-adjusted, matches what physics engine checks)
    seg_lines = []
    for i in range(len(route) - 1):
        mins = get_seg_minutes(route[i], route[i + 1], train_type)
        zone = _STATION_ZONE.get(route[i + 1], "NCR")
        mult = ZONE_WEATHER.get(zone, {}).get("multiplier", 1.0)
        seg_lines.append(f"  {route[i]}→{route[i+1]}: {mins}min  (zone={zone}, ×{mult})")
    seg_table = "\n".join(seg_lines)

    # Compact timetable of the last 5 committed trains for conflict awareness
    if committed_timetable:
        tt_lines = []
        for t in committed_timetable[-5:]:
            first = t["stops"][0]  if t["stops"] else {}
            last  = t["stops"][-1] if t["stops"] else {}
            tt_lines.append(
                f"  {t['id']} ({t['type']}): "
                f"{first.get('station','?')} dep {first.get('departure','?')} → "
                f"{last.get('station','?')} arr {last.get('arrival','?')}"
            )
        tt_str = "\n".join(tt_lines)
    else:
        tt_str = "  (none — you are first train)"

    # Strategic hints embedded in the task
    hints     = task_def.get("strategic_hints", [])
    hints_str = "\n".join(f"  - {h}" for h in hints) if hints else "  (none)"

    # Pre-calculated valid baseline (the model may use it as-is or adjust)
    baseline_json = json.dumps(fallback_stops, indent=2)

    return f"""You are an Indian Railways dispatcher. Schedule train {train_id}.

TASK: {task_def.get('description', '')}

TRAIN TO SCHEDULE:
  ID: {train_id}  |  Type: {train_type}
  Route ({len(route)} stations): {' → '.join(route)}
  Preferred departure from {route[0]}: {departure}
  Min headway: {min_hw}min  |  Dwell at intermediate stops: {min_dwell}–{max_dwell}min

SEGMENT TRAVEL TIMES (weather-adjusted minimums — DO NOT schedule faster than these):
{seg_table}

ALREADY COMMITTED TRAINS (avoid headway/head-on conflicts with these):
{tt_str}

DISPATCHER NOTES FOR THIS TRAIN: {train_def.get('note', 'None.')}

STRATEGIC HINTS:
{hints_str}

CALCULATED BASELINE SCHEDULE (physically valid — use unless you see a conflict):
{baseline_json}

OUTPUT RULES:
- Respond ONLY with raw JSON — no markdown fences, no prose outside the JSON
- If the baseline already avoids conflicts with committed trains, return it unchanged
- If you detect a headway/collision conflict, adjust departure times accordingly
- First stop: "arrival" must be null
- Last stop:  "departure" must be null
- Intermediate stops: both "arrival" and "departure" required (dwell ≥ {min_dwell}min)
- All times in HH:MM 24-hour format
- "day_offset": 0 same day, 1 if the time is past midnight
- "platform": positive integer within the station's platform count

Respond with exactly this structure:
{{
  "action_type": "generate_schedule",
  "reasoning": "<one sentence: which conflict you found and how you fixed it>",
  "schedule_proposal": {{
    "train_id": "{train_id}",
    "train_type": "{train_type}",
    "stops": [ {{"station":"...","arrival":...,"departure":...,"platform":1,"day_offset":0}}, ... ]
  }}
}}"""

# ── 9. LLM CALLER ────────────────────────────────────────────────────────
def call_llm(prompt: str) -> dict:
    system = (
        "You are an expert railway scheduling system. "
        "Your output must be exclusively raw, valid JSON with no markdown formatting."
    )
    try:
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2500,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if model adds them despite instructions
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$",        "", raw, flags=re.MULTILINE)
        # Extract the JSON object (same robust pattern as your working AWS version)
        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(raw)
    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return {}

# ── 10. ACTION PARSER ─────────────────────────────────────────────────────
def parse_action(
    payload:        dict,
    train_id:       str,
    train_type:     str,
    fallback_stops: List[dict],
) -> TrainAction:
    """
    Build a valid TrainAction from LLM output.
    Falls back to the physics-calculated schedule on ANY parse error,
    so we always submit something scoreable rather than a 0.0.
    """
    try:
        proposal_data = payload.get("schedule_proposal", {})
        raw_stops     = proposal_data.get("stops", [])

        if not isinstance(raw_stops, list) or len(raw_stops) < 2:
            raise ValueError(f"Expected ≥2 stops, got {type(raw_stops)}")

        stops    = [StopTime(**s) for s in raw_stops]
        proposal = ScheduleProposal(
            train_id   = proposal_data.get("train_id",   train_id),
            train_type = proposal_data.get("train_type", train_type),
            stops      = stops,
        )
        return TrainAction(
            action_type       = payload.get("action_type", "generate_schedule"),
            reasoning         = payload.get("reasoning",   "LLM-generated schedule"),
            schedule_proposal = proposal,
        )

    except Exception:
        # Physics-calculated fallback — guaranteed valid structure
        stops    = [StopTime(**s) for s in fallback_stops]
        proposal = ScheduleProposal(train_id=train_id, train_type=train_type, stops=stops)
        return TrainAction(
            action_type       = "generate_schedule",
            reasoning         = (
                f"Physics-calculated schedule: {train_id} "
                f"{fallback_stops[0]['station']}→{fallback_stops[-1]['station']} "
                f"dep {fallback_stops[0].get('departure','?')}"
            ),
            schedule_proposal = proposal,
        )

# ── 11. MAIN BENCHMARK RUNNER ─────────────────────────────────────────────
def run_benchmark():
    task_ids = ["easy_task_1", "medium_task_2", "hard_task_3"]

    for task_id in task_ids:
        task_def = TASK_REGISTRY[task_id]
        trains   = task_def["trains"]

        log_start(task=task_id, env="benchmark", model=MODEL_NAME)

        step_count          = 0
        all_rewards:  List[float] = []
        train_scores: List[float] = []
        trains_scheduled    = 0
        committed_timetable: List[dict] = []   # client-side context for conflict prompts
        final_score         = 0.0
        success             = False

        try:
            with MyEnv(base_url=ENV_URL).sync() as client:
                client.reset()   # fresh timetable for this task

                for t_idx, train_def in enumerate(trains):
                    is_last_train = (t_idx == len(trains) - 1)
                    route    = calculate_route(train_def["src"], train_def["dest"])
                    fallback = build_fallback_stops(route, train_def["time"], train_def["type"])

                    for attempt in range(3):
                        step_count     += 1
                        is_last_attempt = (attempt == 2)

                        llm_out    = call_llm(
                            build_prompt(train_def, route, committed_timetable, task_def, fallback)
                        )
                        action     = parse_action(llm_out, train_def["id"], train_def["type"], fallback)
                        action_str = _format_action(action.model_dump(exclude_none=True))

                        try:
                            result = client.step(action)
                            reward = float(result.reward)
                            error  = result.observation.server_feedback if reward < 0.7 else None
                        except Exception as e:
                            reward, error = 0.0, str(e)

                        all_rewards.append(reward)
                        done = is_last_train and (reward >= 0.6 or is_last_attempt)
                        log_step(step_count, action_str, reward, done, error)

                        if reward >= 0.6:
                            train_scores.append(reward)
                            trains_scheduled += 1
                            # Keep committed schedule so next train can avoid conflicts
                            if action.schedule_proposal:
                                committed_timetable.append({
                                    "id":    train_def["id"],
                                    "type":  train_def["type"],
                                    "stops": [s.model_dump() for s in action.schedule_proposal.stops],
                                })
                            break

                        if is_last_attempt:
                            train_scores.append(0.0)

            avg_score   = sum(train_scores) / len(train_scores) if train_scores else 0.0
            final_score = grade_task(task_id, trains_scheduled, len(trains), avg_score)
            success     = final_score >= 0.5

        except Exception as e:
            final_score = 0.0
            print(f"[ERROR] {task_id}: {e}", flush=True)

        log_end(success=success, steps=step_count, score=final_score, rewards=all_rewards)


if __name__ == "__main__":
    # Quick LLM connectivity check before starting benchmark
    try:
        llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        print(f"[INFO] LLM OK: {MODEL_NAME} via {API_BASE_URL}", flush=True)
    except Exception as e:
        print(f"[WARN] LLM ping failed: {e}", flush=True)

    run_benchmark()
