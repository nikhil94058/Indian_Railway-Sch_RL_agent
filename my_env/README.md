---
title: Indian Railway Scheduling Environment
emoji: 🚄
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - railway-scheduling
base_path: /web
---

# Indian Railway Scheduling — RL Environment

A **production-grade OpenEnv environment** for training LLM-based agents to schedule
trains on a realistic 32-station Indian Railways network. The problem mirrors the daily
challenge faced by real railway dispatchers: assigning departure times and platforms to
hundreds of trains while avoiding head-on collisions, headway violations, and platform
conflicts across multiple weather zones.

---

## Real-World Motivation

Indian Railways runs **~13,000 trains daily** across 68,000 km of track. Scheduling
conflicts cause cascading delays that cost billions annually. This environment captures
the core constraint-satisfaction challenge:

- **32 real stations** from New Delhi (NDLS) to Visakhapatnam (VSKP)
- **Multi-zone weather** — Northern fog (1.5× delay), Eastern cyclones (2.0× delay)
- **Mixed fleet** — Rajdhani, Shatabdi, Duronto, Express, Passenger, Freight
- **Real physics** — headway rules (≥15 min), single-track head-on detection, platform capacity

---

## Environment Setup

### Prerequisites

```bash
pip install openenv openai python-dotenv pydantic
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Yes | Model ID (e.g. `meta-llama/Meta-Llama-3.1-70B-Instruct`) |
| `HF_TOKEN` | Yes | HuggingFace token (used as API key) |
| `ENV_URL` | No | Environment server URL (default: `http://localhost:8000`) |

Create a `.env` file:
```
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
HF_TOKEN=hf_...
ENV_URL=http://localhost:8000
```

### Quick Start

```bash
# 1. Start the environment server
uvicorn my_env.server.app:app --host 0.0.0.0 --port 8000

# 2. Run the agent (from project root)
python inference.py
```

---

## Observation Space

The agent receives a `TrainObservation` at every step:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current episode identifier |
| `goal` | `str` | Task description and active task name |
| `network_summary` | `dict` | Full station list (platforms, zone) + track list (capacity, travel time) |
| `timetable` | `List[TrainInfo]` | All trains committed to the timetable so far |
| `new_train_request` | `TrainRequest` | The train to be scheduled next (origin, dest, type, preferred departure) |
| `step_count` | `int` | Steps taken in current episode |
| `max_steps` | `int` | Episode step limit |
| `reward` | `float` | Physics score for the last action (0.0–1.0) |
| `done` | `bool` | Episode complete flag |
| `server_feedback` | `str` | Human-readable error message from the physics engine |
| `metadata.knowledge_base` | `dict` | Learned patterns: platform preferences, congestion windows, past failures/successes |

### TrainInfo (timetable entry)
```json
{
  "id": "RAJ-05",
  "name": "Service RAJ-05",
  "type": "superfast",
  "days": ["Daily"],
  "stops": [
    {"station": "NDLS", "arrival": null, "departure": "06:00", "platform": 3, "day_offset": 0},
    {"station": "CNB",  "arrival": "09:10", "departure": "09:15", "platform": 2, "day_offset": 0},
    {"station": "HWH",  "arrival": "22:30", "departure": null, "platform": 7, "day_offset": 0}
  ]
}
```

---

## Action Space

The agent sends a `TrainAction`:

| Field | Type | Description |
|---|---|---|
| `action_type` | `Literal` | One of: `generate_schedule`, `detect_conflicts`, `suggest_slot`, `noop` |
| `reasoning` | `str` | Chain-of-thought explanation (required — enables explainability scoring) |
| `schedule_proposal` | `ScheduleProposal` | Proposed stop-times for the new train |
| `conflict_flags` | `List[ConflictFlag]` | Optional: conflicts the agent detected |

### Example Action
```json
{
  "action_type": "generate_schedule",
  "reasoning": "EXP-20 runs CNB→PNBE on the same track as EXP-19 at 13:30. I delayed EXP-20 departure by 20 min to maintain headway.",
  "schedule_proposal": {
    "train_id": "EXP-20",
    "stops": [
      {"station": "PNBE", "arrival": null,    "departure": "13:55", "platform": 4, "day_offset": 0},
      {"station": "CNB",  "arrival": "17:30",  "departure": null,    "platform": 6, "day_offset": 0}
    ]
  }
}
```

---

## Reward Function

The physics engine returns a **continuous score in [0.0, 1.0]** — no sparse binary rewards:

```
score = (conflict_free × 0.40) + (journey_time × 0.20) + (platform_valid × 0.15)
      + (headway_quality × 0.12) + (priority_respected × 0.08) + (dwell_valid × 0.05)
```

| Component | Weight | What it measures |
|---|---|---|
| `conflict_free` | 40% | Binary — any HEAD_ON, REAR_END, OVERTAKE, PLATFORM_CLASH, or TRACK_FULL sets this to 0 |
| `journey_time` | 20% | Continuous — fraction of segments physically achievable under weather conditions |
| `platform_valid` | 15% | Binary — any platform number exceeding station capacity sets this to 0 |
| `headway_quality` | 12% | Continuous — actual headway gap as fraction of required minimum (rewards wider margins) |
| `priority_respected` | 8% | Continuous — fraction of priority interactions not violated |
| `dwell_valid` | 5% | Continuous — fraction of intermediate stops with dwell within type-specific min/max |

Scores ≥ 0.6 commit the train to the permanent timetable. Lower scores are recorded
as failures in the knowledge base for the agent to learn from.

---

## Tasks & Graders

Three independent tasks of increasing difficulty. Each returns a grade via:

```
grade = 0.5 × (trains_scheduled / trains_total) + 0.5 × avg_physics_score
```

Each task is designed so a naive agent (raw skeleton times, no conflict reasoning) **will fail**.
A learning agent that reads errors, uses the knowledge base, and applies cascade delay logic **can succeed**.

---

### Task 1 — Easy (`easy_task_1`)
**"Operation Fogbreaker: The GZB Convergence Trap"**

**Scenario**: Winter morning NDLS departure cluster under IMD Dense Fog Alert (NR zone, 1.5× travel times).

| Train | Type | Route | Departure |
|---|---|---|---|
| PASS-01 | Passenger | NDLS→CNB | 04:50 |
| EXP-04  | Express   | NDLS→PNBE | 05:40 |
| RAJ-05  | Rajdhani  | NDLS→HWH | 06:00 |

**The engineered trap**: All 3 trains travel NDLS→GZB under identical fog conditions (67.5 min each), arriving GZB simultaneously. Rajdhani's 2-min dwell vs Express's 5-min dwell causes RAJ-05 to depart GZB **before** EXP-04 — creating a 3-min headway where 15 min is required → **REAR-END** on GZB→ALJN 2-track.

**Challenge types**: `fog_speed_compliance` · `headway_convergence_trap` · `type_specific_dwell` · `priority_order`

- Max steps: **8** | Baseline score: **0.88**

---

### Task 2 — Medium (`medium_task_2`)
**"Operation Mughalsarai: The Bihar Bottleneck Breakdown"**

**Scenario**: DDU Junction (formerly Mughalsarai — busiest marshalling yard in Asia) under ECR Heavy Rain (1.2×). Five trains converge with **three simultaneous conflict types**.

| Train | Type | Route | Departure | Conflict Role |
|---|---|---|---|---|
| FR-24  | Freight  | DDU→ASN  | 11:30 | Blocks Rajdhani (10min gap, need 25min) |
| RAJ-27 | Rajdhani | DDU→HWH  | 11:40 | Highest priority, rear-ended by Freight |
| EXP-19 | Express  | CNB→PNBE | 13:30 | HEAD-ON with EXP-20 on 2-track corridor |
| EXP-20 | Express  | PNBE→CNB | 13:35 | HEAD-ON with EXP-19 (opposite direction) |
| PASS-21| Passenger| ARA→DDU  | 13:40 | Platform saturation at ARA (4 platforms) |

**Three simultaneous failures**:
1. **REAR-END + PRIORITY**: FR-24 (Freight) and RAJ-27 (Rajdhani) depart DDU 10 min apart — need 25 min
2. **HEAD-ON**: EXP-19 and EXP-20 converging on DDU↔PNBE 2-track, 5 min apart
3. **PLATFORM SATURATION**: EXP-19, EXP-20, PASS-21 all need ARA platforms (only 4 available) simultaneously

**Challenge types**: `rear_end_prevention` · `priority_clearance` · `head_on_resolution` · `platform_saturation` · `rain_speed_compliance`

- Max steps: **15** | Baseline score: **0.67**

---

### Task 3 — Hard (`hard_task_3`)
**"Operation Cyclone Exodus: East Coast Multi-Crisis"**

**Scenario**: Inspired by Cyclone Phailin (Oct 2013, Category 5) — Indian Railways evacuated 980,000 people in 48 hours. 9 trains must run simultaneously across all 7 weather zones. **Every physics engine check fires**.

| Train | Type | Route | Departure | Key Challenge |
|---|---|---|---|---|
| RAJ-31  | Rajdhani  | NDLS→BBS | 22:00 | Overnight, cyclone zone arrival, day_offset=1 |
| DUR-33  | Duronto   | NDLS→HWH | 23:00 | Follows RAJ-31, midnight crossing |
| EXP-12  | Express   | HWH→VSKP | 08:00 | 2× cyclone zone, ~24hr journey |
| DUR-14  | Duronto   | HWH→BBS  | 08:30 | Must not be rear-ended by FR-13 |
| FR-13   | Freight   | ASN→BHC  | 07:30 | Lowest priority, yields to all; rear-end risk |
| PASS-15 | Passenger | SRC→KGP  | 09:00 | Cyclone zone, KGP platform saturation |
| EXP-16  | Express   | KGP→CTC  | 09:15 | Deep cyclone zone, head-on with EXP-36 |
| EXP-36  | Express   | PNBE→NDLS | 07:00 | Going WEST — head-on risk with eastbound trains |
| SHAT-39 | Shatabdi  | LKO→NDLS | 08:00 | Fastest westbound, SHAT overtakes EXP-36 in fog |

**All 6 collision types triggered simultaneously**:

| Collision | Trains Involved | Segment |
|---|---|---|
| HEAD-ON | EXP-16 ↔ EXP-36 | KGP↔CTC 2-track |
| REAR-END | FR-13 chasing DUR-14 | KGP→BLS |
| OVERTAKE | SHAT-39 catches EXP-36 | TDL→GZB 2-track |
| PLATFORM CLASH | EXP-12, DUR-14 | HWH + KGP |
| TRACK FULL | 3 eastbound trains | KGP→BLS 2-track |
| SPEED VIOLATION | Any train ignoring 2× | SER/ECoR cyclone |

**Challenge types**: `head_on_multi_segment` · `rear_end_freight_premium` · `overtake_fog_segment` · `platform_saturation_hwh` · `platform_saturation_kgp` · `track_full_cyclone_zone` · `midnight_crossing` · `weather_cascade` · `priority_chain` · `cyclone_speed_enforcement`

- Max steps: **30** | Baseline score: **0.41**

---

## Knowledge Base & Agent Learning

The environment maintains a persistent `knowledge_base.json` that accumulates across
runs. The agent receives a **compact summary** in `metadata.knowledge_base` each step:

```json
{
  "platform_preferences": {"CNB": {"Express": 6, "Passenger": 3}},
  "congestion_windows":   {"NDLS-GZB": ["06:00", "07:30"]},
  "recent_failures": [...],
  "recent_successes": [...]
}
```

`inference.py` additionally applies **smart timetable filtering**: only trains sharing
≥1 track segment with the current train's route are included in the prompt. This reduces
token consumption by up to 90% as the timetable grows.

---

## Stdout Protocol

`inference.py` emits structured logs to stdout:

```
[START] task_id=easy_task_1 difficulty=easy trains=1
[STEP]  step=1 train=PASS-01 attempt=1 reward=0.920 feedback='Perfect Schedule.'
[END]   task_id=easy_task_1 trains_scheduled=1/1 avg_score=0.920 final_grade=0.920
```

---

## Architecture

```
inference.py  (entry point)
├── OpenAI-compatible LLM (API_BASE_URL + MODEL_NAME + HF_TOKEN)
├── Smart timetable filter  →  only relevant trains sent to LLM
├── KB hint builder         →  platform prefs + congestion windows
└── MyEnv client  ──────────────────►  FastAPI server (port 8000)
                                            └── TrainSchedulingEnv
                                                ├── PhysicsEngine  (collision detection)
                                                ├── DisruptionEngine (random delays)
                                                └── DataManager    (KB persistence)
```

---

## Project Structure

```
.
├── inference.py                    # Hackathon entry point (OpenAI client + [START/STEP/END] logs)
└── my_env/
    ├── client.py                   # MyEnv OpenEnv client
    ├── models.py                   # Pydantic models (Action / Observation / Reward)
    ├── data/
    │   ├── network.py              # 32-station network definition
    │   └── knowledge_base.json     # Persistent learned patterns (auto-generated)
    └── server/
        ├── app.py                  # FastAPI application
        └── my_env_environment.py   # TrainSchedulingEnv + TASK_REGISTRY + graders
```
