"""
inference.py  —  Indian Railway Scheduling RL Agent
====================================================
Hackathon entry point. Runs the LLM-based dispatcher agent against
the OpenEnv railway scheduling environment.

Required environment variables
-------------------------------
  API_BASE_URL  : OpenAI-compatible endpoint (e.g. HuggingFace Inference)
  MODEL_NAME    : Model identifier (e.g. "meta-llama/Meta-Llama-3.1-70B-Instruct")
  HF_TOKEN      : HuggingFace token used as the OpenAI API key
  ENV_URL       : (optional) Environment server URL, default http://localhost:8000
  DEBUG         : (optional) Set to True/1 to print raw LLM prompts and responses
"""

import os
import sys
import json
import heapq
import re
from pathlib import Path
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Validate env vars ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
MODEL_NAME   = os.environ.get("MODEL_NAME", "").strip()
HF_TOKEN     = os.environ.get("HF_TOKEN", "").strip()
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000").strip()
DEBUG_MODE   = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    print("[ERROR] Missing required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN")
    sys.exit(1)

llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Project path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from my_env.client import MyEnv
from my_env.models import TrainAction
from my_env.data.network import data
from my_env.data.config import ZONE_WEATHER          # single source of truth
from my_env.server.my_env_environment import TASK_REGISTRY, grade_task

# ═══════════════════════════════════════════════════════════════════════════
# NETWORK SETUP
# ═══════════════════════════════════════════════════════════════════════════
STATIONS = {
    s["id"]: {"name": s["name"], "platforms": s["platform_count"], "zone": s["zone"]}
    for s in data["stations"]
}

GRAPH: dict = {}
TRACKS: dict = {}
for _t in data["tracks"]:
    u, v, w, c = _t["from"], _t["to"], _t["travel_minutes"], _t.get("track_count", 2)
    GRAPH.setdefault(u, []).append((v, w))
    GRAPH.setdefault(v, []).append((u, w))
    TRACKS.setdefault(u, {})[v] = {"time": w, "capacity": c}
    TRACKS.setdefault(v, {})[u] = {"time": w, "capacity": c}

# ═══════════════════════════════════════════════════════════════════════════
# PATHFINDING & ROUTE PHYSICS
# ═══════════════════════════════════════════════════════════════════════════
def find_fastest_route(start: str, end: str) -> list:
    queue = [(0, start, [])]
    visited: set = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return path
        for neighbor, weight in GRAPH.get(node, []):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path))
    return []


def calculate_route(src: str, dest: str, via: str = None) -> list:
    if not via:
        return find_fastest_route(src, dest)
    leg1 = find_fastest_route(src, via)
    leg2 = find_fastest_route(via, dest)
    return (leg1[:-1] + leg2) if leg1 and leg2 else []


def get_travel_skeleton(route: list, start_time: str, train_type: str) -> tuple[list, str]:
    """Returns stop-by-stop timing skeleton and weather summary string."""
    travel_data = []
    current_time = datetime.strptime(start_time, "%H:%M")
    day_offset = 0
    weather_log: list = []

    for i, station_id in enumerate(route):
        info = STATIONS.get(station_id, {})
        zone = info.get("zone", "NR")
        max_plats = info.get("platforms", 5)
        weather = ZONE_WEATHER.get(zone, {"condition": "Clear", "multiplier": 1.0})

        label = f"{zone}: {weather['condition']}"
        if label not in weather_log:
            weather_log.append(label)

        # Determine activity & dwell
        if i == 0:
            activity, dwell = "ORIGIN", 0
        elif i == len(route) - 1:
            activity, dwell = "DESTINATION", 0
        else:
            is_junction = info.get("is_junction", False)
            if train_type in ("Rajdhani", "Shatabdi", "Duronto"):
                activity = "MAJOR_STOP" if is_junction else "THROUGH_PASS"
                dwell = 5 if activity == "MAJOR_STOP" else 1
            elif train_type == "Express":
                activity = "MAJOR_STOP" if max_plats >= 4 else "THROUGH_PASS"
                dwell = 10 if activity == "MAJOR_STOP" else 1
            elif train_type == "Passenger":
                activity, dwell = "MAJOR_STOP", 15
            else:
                activity = "MAJOR_STOP" if is_junction else "THROUGH_PASS"
                dwell = 30 if activity == "MAJOR_STOP" else 1

        if i > 0:
            prev = route[i - 1]
            base_mins = TRACKS[prev][station_id]["time"]
            actual_mins = int(base_mins * weather["multiplier"])
            current_time += timedelta(minutes=actual_mins)
            if current_time.day > 1:
                day_offset += 1
                current_time = current_time.replace(day=1)

        arr_str = current_time.strftime("%H:%M") if i > 0 else None

        if 0 < i < len(route) - 1:
            current_time += timedelta(minutes=dwell)
            if current_time.day > 1:
                day_offset += 1
                current_time = current_time.replace(day=1)

        dep_str = current_time.strftime("%H:%M") if i < len(route) - 1 else None
        next_cap = TRACKS[station_id][route[i + 1]]["capacity"] if i < len(route) - 1 else "N/A"

        travel_data.append({
            "station": station_id,
            "MAX_PLATFORMS": max_plats,
            "NEXT_TRACK_CAPACITY": next_cap,
            "activity": activity,
            "earliest_arrival": arr_str,
            "earliest_departure": dep_str,
            "day_offset": day_offset,
            "ASSIGN_PLATFORM_HERE": "integer 1..MAX_PLATFORMS",
        })

    return travel_data, " | ".join(weather_log)


# ═══════════════════════════════════════════════════════════════════════════
# SMART CONTEXT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════
def filter_relevant_timetable(route: list, full_timetable: list) -> list:
    """
    Returns only trains that share ≥1 track segment with the current route.
    Drastically cuts token usage as the timetable grows.
    """
    route_pairs: set = set()
    for i in range(len(route) - 1):
        pair = tuple(sorted([route[i], route[i + 1]]))
        route_pairs.add(pair)

    relevant = []
    for train in full_timetable:
        stops = train.get("stops", []) if isinstance(train, dict) else train.stops
        for j in range(len(stops) - 1):
            s1 = stops[j]["station"] if isinstance(stops[j], dict) else stops[j].station
            s2 = stops[j + 1]["station"] if isinstance(stops[j + 1], dict) else stops[j + 1].station
            if tuple(sorted([s1, s2])) in route_pairs:
                relevant.append(train if isinstance(train, dict) else train.model_dump())
                break

    return relevant


def build_kb_hint(kb: dict, route: list) -> str:
    """
    Summarise relevant KB entries into a compact natural-language hint
    to guide the LLM without consuming excessive tokens.
    """
    hints: list = []

    # Platform preferences for stations on this route
    prefs = kb.get("platform_preferences", {})
    for station in route:
        if station in prefs:
            for train_type, plat in prefs[station].items():
                hints.append(f"  - {station}: {train_type} → platform {plat} worked before.")

    # Congestion windows on relevant track segments
    cw = kb.get("congestion_windows", {})
    for i in range(len(route) - 1):
        seg_key = f"{route[i]}-{route[i+1]}"
        rev_key = f"{route[i+1]}-{route[i]}"
        for key in (seg_key, rev_key):
            if key in cw:
                times = ", ".join(cw[key])
                hints.append(f"  - Track {key} congested around: {times}. Adjust departure.")

    # Recent failures (last 3)
    for f in kb.get("recent_failures", [])[-3:]:
        hints.append(f"  - PAST FAIL [{f['train_id']}]: {f['error'][:120]}")

    return "\n".join(hints) if hints else "  (no prior knowledge yet)"


# ═══════════════════════════════════════════════════════════════════════════
# LLM CALL (WITH DEBUGGING)
# ═══════════════════════════════════════════════════════════════════════════
def call_llm(prompt: str) -> dict:
    """Call the OpenAI-compatible endpoint and parse JSON response."""
    
    if DEBUG_MODE:
        print("\n\033[96m=== [DEBUG] PROMPT SENT TO LLM ===\033[0m")
        print(prompt)
        print("\033[96m===================================\033[0m\n")

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Indian Railway dispatcher. "
                        "Output ONLY raw valid JSON — no markdown, no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        if DEBUG_MODE:
            print("\n\033[93m=== [DEBUG] RAW LLM RESPONSE ===\033[0m")
            print(raw)
            print("\033[93m================================\033[0m\n")

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        return json.loads(match.group(1) if match else raw)

    except json.JSONDecodeError as e:
        return {"error": "JSON parse failed", "details": str(e)}
    except Exception as e:
        return {"error": "LLM call failed", "details": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE TASK RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def run_task(task_id: str) -> float:
    """
    Run one discrete task. Returns the final grade (0.0–1.0).
    Emits [START], [STEP], [END] to stdout as required by the spec.
    """
    task_def = TASK_REGISTRY[task_id]
    trains = task_def["trains"]
    max_attempts_per_train = 10

    print(f"[START] task_id={task_id} difficulty={task_def['difficulty']} trains={len(trains)}")

    trains_scheduled = 0
    scores: list = []
    global_step = 0

    with MyEnv(base_url=ENV_URL).sync() as client:
        obs = client.reset().observation

        for train_def in trains:
            route = calculate_route(
                train_def["src"],
                train_def["dest"],
                train_def.get("via"),
            )
            if not route:
                print(f"[STEP] step={global_step} train={train_def['id']} status=NO_ROUTE score=0.0")
                continue

            skeleton, weather_log = get_travel_skeleton(route, train_def["time"], train_def["type"])

            # Filter timetable to only relevant trains (token saver)
            relevant_tt = filter_relevant_timetable(route, obs.timetable)

            # Pull learned knowledge for this route
            kb = obs.metadata.get("knowledge_base", {})
            kb_hint = build_kb_hint(kb, route)

            success = False
            last_error = "None"

            # Build dispatcher context from task metadata
            train_note      = train_def.get("note", "")
            task_hints      = task_def.get("strategic_hints", [])
            challenge_types = task_def.get("challenge_types", [])

            # Only show hints relevant to this train (contain its ID or general)
            relevant_hints = [
                h for h in task_hints
                if train_def["id"] in h or not any(
                    t["id"] in h for t in trains if t["id"] != train_def["id"]
                )
            ]
            hints_text = "\n".join(f"  • {h}" for h in relevant_hints[:6]) or "  (none)"
            challenges_text = ", ".join(challenge_types)

            for attempt in range(1, max_attempts_per_train + 1):
                global_step += 1

                prompt = f"""
[INDIAN RAILWAY DISPATCH — {task_def['difficulty'].upper()} SCENARIO]
[Train {train_def['id']} | Type: {train_def['type']} | {train_def['src']}→{train_def['dest']}]

SCENARIO: {task_def['scenario_context']}

DISPATCHER NOTE FOR THIS TRAIN:
  {train_note}

ACTIVE CHALLENGES IN THIS SCENARIO: {challenges_text}

STRATEGIC HINTS (study before scheduling):
{hints_text}

WEATHER EN ROUTE: {weather_log}

LEARNED KNOWLEDGE FROM PAST RUNS (apply these patterns):
{kb_hint}

ROUTE SKELETON (earliest physically possible times — you MAY delay further):
{json.dumps(skeleton, indent=2)}

TRAINS ALREADY ON SHARED TRACK SEGMENTS (conflict candidates):
{json.dumps(relevant_tt, indent=2)}

LAST SERVER ERROR TO FIX (if any): {last_error}

PHYSICS ENGINE RULES (all enforced — violations score 0 on that component):
1. PLATFORM     : integer 1..MAX_PLATFORMS for each station. No exceptions.
2. HEADWAY      : same-direction departure gap ≥ type minimum:
                  Rajdhani/Shatabdi/Duronto ≥10min | Express ≥15min | Passenger ≥20min | Freight ≥25min
                  Required gap = max(your_type_min, their_type_min).
3. HEAD-ON      : opposite trains on same 2-track segment simultaneously → CRASH (conflict_free=0).
4. OVERTAKE     : departing after a slower train but arriving before it on single track → CRASH.
5. WEATHER SPEED: proposed segment time ≥ base_time × multiplier × 0.90.
                  NR Fog=1.5× | ECR Rain=1.2× | SER/ECoR Cyclone=2.0× | Others=1.0×
                  Violating this by >10% sets conflict_free=0.
6. DWELL TIME   : intermediate stop dwell ≥ type_min AND ≤ type_max.
                  Rajdhani:2–10min | Express:5–20min | Passenger:10–30min | Freight:15–60min
7. DAY OFFSET   : for overnight trains, stops after midnight need day_offset=1.
8. CASCADE RULE : if you delay a departure, add the same delay to ALL subsequent arrivals/departures.

OUTPUT — raw JSON only, no markdown:
{{
  "action_type": "generate_schedule",
  "reasoning": "<think step by step: identify conflicts, state which rules apply, explain every delay>",
  "schedule_proposal": {{
    "train_id": "{train_def['id']}",
    "train_type": "{train_def['type']}",
    "stops": [
      {{"station": "NDLS", "arrival": null, "departure": "HH:MM", "platform": 2, "day_offset": 0}},
      {{"station": "GZB",  "arrival": "HH:MM", "departure": "HH:MM", "platform": 3, "day_offset": 0}}
    ]
  }}
}}
"""

                llm_response = call_llm(prompt)

                if "error" in llm_response:
                    print(f"[STEP] step={global_step} train={train_def['id']} attempt={attempt} status=LLM_ERROR score=0.0")
                    last_error = llm_response.get("details", "LLM error")
                    continue

                try:
                    action = TrainAction(**llm_response)
                    result = client.step(action)
                    obs = result.observation
                    reward = result.reward

                    print(
                        f"[STEP] step={global_step} train={train_def['id']} "
                        f"attempt={attempt} reward={reward:.3f} "
                        f"feedback={obs.server_feedback[:80]!r}"
                    )

                    if reward >= 0.6:
                        scores.append(reward)
                        trains_scheduled += 1
                        success = True

                        # Refresh relevant timetable and KB after success
                        relevant_tt = filter_relevant_timetable(route, obs.timetable)
                        kb = obs.metadata.get("knowledge_base", {})
                        kb_hint = build_kb_hint(kb, route)
                        break
                    else:
                        last_error = obs.server_feedback or "Physics rejection."

                except Exception as e:
                    print(f"[STEP] step={global_step} train={train_def['id']} attempt={attempt} status=PARSE_ERROR score=0.0")
                    last_error = str(e)

            if not success:
                scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    final_grade = grade_task(task_id, trains_scheduled, len(trains), avg_score)

    print(
        f"[END] task_id={task_id} trains_scheduled={trains_scheduled}/{len(trains)} "
        f"avg_score={avg_score:.3f} final_grade={final_grade:.3f}"
    )
    return final_grade


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """
    Run all three tasks in order of difficulty and report overall performance.
    """
    task_ids = ["easy_task_1", "medium_task_2", "hard_task_3"]
    results: dict = {}

    for task_id in task_ids:
        grade = run_task(task_id)
        results[task_id] = grade

    overall = sum(results.values()) / len(results)
    print("\n══════════════════════════════════════")
    print("FINAL SCORES")
    print("══════════════════════════════════════")
    for tid, score in results.items():
        diff = TASK_REGISTRY[tid]["difficulty"].upper()
        print(f"  {diff:<8} {tid}: {score:.3f}")
    print(f"  OVERALL : {overall:.3f}")
    print("══════════════════════════════════════")


if __name__ == "__main__":
    main()