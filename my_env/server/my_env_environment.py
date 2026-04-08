# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import uuid
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from openenv.core.env_server import Environment, State
from ..models import (
    TrainAction, TrainObservation, TrainInfo,
    StopTime, TrainRequest, TrainReward, RewardBreakdown, ScheduleProposal,
)
from ..data.config import (
    ZONE_WEATHER, TRAIN_SPECS, DEFAULT_TRAIN_SPEC,
    SCORE_WEIGHTS, PREFIX_TO_TYPE, infer_train_type,
)


# ===================================================================
# TASK REGISTRY  —  3 world-class operational scenarios
# ===================================================================
#
# Design principles:
#   - Each task maps to a real documented Indian Railways operational challenge
#   - Difficulty is engineered: a naive agent WILL fail; a learning agent CAN succeed
#   - Every task contains multiple overlapping conflict types, not just one
#   - Trains include dispatcher notes — the same context real operators use
#   - validation_criteria lists what the grader specifically checks
#
TASK_REGISTRY: Dict[str, Dict] = {

    # ─────────────────────────────────────────────────────────────────
    # TASK 1 — EASY
    # "Operation Fogbreaker: The GZB Convergence Trap"
    #
    # Real scenario: Every winter morning (Dec–Jan) at NDLS, Dense Fog
    # (IMD Category RED, visibility < 50 m) triggers speed restrictions
    # across the Northern Railway zone. A cluster of 3 trains — a slow
    # Passenger, a medium Express, and a fast Rajdhani — must depart
    # within a 70-minute window. The 4-track NDLS–GZB section absorbs
    # them safely. The trap is at GZB onward (2-track): because all 3
    # trains travel NDLS→GZB under the same fog multiplier (1.5×) they
    # arrive at GZB within minutes of each other. The Rajdhani's shorter
    # minimum dwell (2 min vs Express 5 min) causes it to depart GZB
    # BEFORE the Express despite leaving NDLS later — creating a
    # 3-minute headway where 15 minutes is required.
    #
    # A naive agent uses the raw skeleton and misses this convergence.
    # A smart agent delays the Rajdhani's NDLS departure or manages
    # GZB dwell to guarantee ≥15 min separation on the 2-track onward.
    # ─────────────────────────────────────────────────────────────────
    "easy_task_1": {
        "difficulty": "easy",

        "scenario_context": (
            "NDLS Winter Fog Operations — 04:50–06:20 window. "
            "IMD Dense Fog Alert active for Northern Railway zone (NR). "
            "All trains on NDLS–GZB segment restricted to 60 km/h. "
            "Beyond GZB the NCR zone is clear but lines are 2-track only."
        ),

        "description": (
            "Schedule 3 trains in the NDLS dawn departure window under NR Dense Fog. "
            "The fog multiplier (1.5×) causes all trains to arrive GZB simultaneously. "
            "Rajdhani's 2-min dwell makes it depart GZB BEFORE the Express — "
            "creating a headway violation on the 2-track GZB→ALJN segment. "
            "Agent must detect this convergence and delay the Rajdhani's "
            "NDLS departure to guarantee ≥15 min GZB-departure gap."
        ),

        "challenge_types": [
            "fog_speed_compliance",     # All times must respect 1.5× NR multiplier
            "headway_convergence_trap", # Fog equalises arrival times at GZB
            "type_specific_dwell",      # Rajdhani 2min vs Express 5min dwell matters
            "priority_order",           # Rajdhani (priority 1) needs clear path
            "platform_assignment",      # NDLS 16 plats, GZB 6 plats — must not clash
        ],

        "strategic_hints": [
            "Under fog (1.5×), NDLS→GZB takes 67.5 min for ALL train types.",
            "If EXP departs NDLS at 05:40, it arrives GZB at 07:07 and departs at 07:12 (5-min dwell).",
            "If RAJ departs NDLS at 06:00, it also arrives GZB at 07:07 and departs at 07:09 (2-min dwell).",
            "RAJ departs GZB at 07:09, EXP at 07:12 → gap is only 3 min. Required: 15 min. REAR-END!",
            "Fix: delay RAJ NDLS departure to ≥06:20 so GZB departure gap from EXP exceeds 15 min.",
            "Alternatively: increase RAJ's GZB dwell time (but Rajdhani max_dwell is 10 min).",
        ],

        "trains": [
            {
                "id": "PASS-01", "type": "Passenger",
                "time": "04:50",  "src": "NDLS", "dest": "CNB",
                "note": (
                    "Slowest train. Sets baseline congestion on main line. "
                    "Departs earliest — give it platform 1 at NDLS. "
                    "Min headway 20 min — all trains behind must respect this."
                ),
            },
            {
                "id": "EXP-04", "type": "Express",
                "time": "05:40", "src": "NDLS", "dest": "PNBE",
                "note": (
                    "50 min after PASS-01. Headway at NDLS: 50 min > 20 min required ✓. "
                    "But watch GZB convergence with RAJ-05. "
                    "Use a different platform from PASS-01 at shared stops."
                ),
            },
            {
                "id": "RAJ-05", "type": "Rajdhani",
                "time": "06:00", "src": "NDLS", "dest": "HWH",
                "note": (
                    "NDLS departure is 20 min after EXP-04 — looks safe. "
                    "TRAP: Due to 2-min Rajdhani dwell vs 5-min Express dwell at GZB, "
                    "RAJ-05 departs GZB only 3 min after EXP-04. "
                    "You MUST delay RAJ-05 NDLS departure to ≥06:20 to fix this. "
                    "Highest priority train — must have a clean path on GZB→ALJN."
                ),
            },
        ],

        "validation_criteria": [
            "No REAR-END violation on GZB→ALJN (require ≥15 min departure gap)",
            "No SPEED VIOLATION — all segment times respect NR fog 1.5× floor",
            "All platform numbers within station capacity",
            "Priority order maintained: RAJ gets clean path behind (not chasing) EXP",
        ],

        "max_steps":      8,
        "baseline_score": 0.88,
    },


    # ─────────────────────────────────────────────────────────────────
    # TASK 2 — MEDIUM
    # "Operation Mughalsarai: The Bihar Bottleneck Breakdown"
    #
    # Real scenario: DDU (Pt. Deen Dayal Upadhyaya Jn, formerly
    # Mughalsarai) is the single busiest marshalling yard in Asia.
    # During a peak congestion event (similar to the 2016 Indore–Patna
    # Express disaster zone), 5 trains converge on the DDU–ARA–PNBE
    # corridor simultaneously with 3 independent conflict types:
    #
    #   1. REAR-END + PRIORITY VIOLATION:
    #      Freight (FR-24, DDU→ASN) and Rajdhani (RAJ-27, DDU→HWH)
    #      depart DDU only 10 min apart (same direction). Required
    #      headway = max(Freight 25min, Rajdhani 10min) = 25 min.
    #      Freight MUST be delayed or Rajdhani given 25-min clearance.
    #
    #   2. HEAD-ON COLLISION:
    #      EXP-19 (CNB→PNBE, going EAST) and EXP-20 (PNBE→CNB, going
    #      WEST) depart only 5 min apart from opposite ends. On the
    #      2-track DDU–BXR–ARA–PNBE corridor, both will be on the same
    #      segment simultaneously → fatal HEAD-ON. One must wait ≥ the
    #      other's full segment transit time before departing.
    #
    #   3. PLATFORM SATURATION at ARA Junction:
    #      ARA has only 4 platforms. EXP-19 (passing EAST through ARA
    #      ~16:30), EXP-20 (passing WEST through ARA ~15:45), and
    #      PASS-21 (originating at ARA 13:40) all need platforms at
    #      nearly the same time. Agent must stagger assignments.
    #
    # ECR zone (Heavy Rain, 1.2× travel times) on all eastern segments.
    # ─────────────────────────────────────────────────────────────────
    "medium_task_2": {
        "difficulty": "medium",

        "scenario_context": (
            "DDU–PNBE Corridor, ECR Zone, Heavy Rain (1.2× travel times). "
            "ARA Junction: 4 platforms, major interchange node. "
            "Peak midday window 11:30–14:00: 5 trains converging. "
            "Freight must clear the line before Rajdhani departs DDU."
        ),

        "description": (
            "Schedule 5 trains on the congested DDU–PNBE corridor under ECR Heavy Rain. "
            "Three simultaneous conflict types: "
            "(1) Freight FR-24 and Rajdhani RAJ-27 depart DDU only 10 min apart — "
            "REAR-END + PRIORITY VIOLATION (need 25 min gap). "
            "(2) Express EXP-19 and EXP-20 are running head-on — both on the "
            "DDU↔PNBE 2-track corridor within 5 min of each other. "
            "(3) ARA Junction has only 4 platforms — EXP-19, EXP-20, and PASS-21 "
            "all need platforms at ARA simultaneously."
        ),

        "challenge_types": [
            "rear_end_prevention",      # FR-24 + RAJ-27: need 25min gap
            "priority_clearance",       # Freight must yield to Rajdhani
            "head_on_resolution",       # EXP-19 vs EXP-20 on 2-track
            "platform_saturation",      # ARA 4-platform limit with 3 trains
            "rain_speed_compliance",    # ECR 1.2× multiplier on eastern segments
        ],

        "strategic_hints": [
            "FR-24 departs DDU at 11:30, RAJ-27 at 11:40 — same direction. Gap=10min < required 25min.",
            "Fix option A: Delay FR-24 to 10:30 so RAJ-27 has 70min clearance behind it.",
            "Fix option B: Delay RAJ-27 to 12:00 so gap from FR-24 is 30min > 25min.",
            "EXP-19 (CNB→PNBE) and EXP-20 (PNBE→CNB) are head-on — delay one by ≥3 hours.",
            "ARA has 4 platforms. Assign different platforms to EXP-19, EXP-20, PASS-21.",
            "Under rain (1.2×), DDU→BXR takes 70×1.2=84 min instead of 70 min — factor this in.",
        ],

        "trains": [
            {
                "id": "FR-24", "type": "Freight",
                "time": "11:30", "src": "DDU", "dest": "ASN",
                "note": (
                    "Heavy goods train, priority 7 (LOWEST). "
                    "Departs DDU 10 min before RAJ-27 — MUST be moved earlier "
                    "to give Rajdhani 25-min clear path. "
                    "ECR rain zone: each segment takes 1.2× longer. "
                    "Min headway from any passenger train: 25 min."
                ),
            },
            {
                "id": "RAJ-27", "type": "Rajdhani",
                "time": "11:40", "src": "DDU", "dest": "HWH",
                "note": (
                    "Highest priority train (1). Cannot be delayed beyond 12:15 "
                    "without causing cascading timetable issues downstream. "
                    "Requires 25-min headway from FR-24 (Freight's min_headway=25). "
                    "Use only the fastest platforms at DDU (platform 1 or 2)."
                ),
            },
            {
                "id": "EXP-19", "type": "Express",
                "time": "13:30", "src": "CNB", "dest": "PNBE",
                "note": (
                    "Going EAST through DDU, BXR, ARA, DNR to PNBE. "
                    "EXP-20 is coming the opposite way — HEAD-ON risk. "
                    "You must ensure EXP-19 and EXP-20 are NOT on the same "
                    "2-track segment simultaneously. Stagger by ≥ segment transit time."
                ),
            },
            {
                "id": "EXP-20", "type": "Express",
                "time": "13:35", "src": "PNBE", "dest": "CNB",
                "note": (
                    "Going WEST — OPPOSITE direction to EXP-19. "
                    "Only 5 min departure gap. They WILL crash on DDU–BXR 2-track. "
                    "Delay this train by at least 3–4 hours, OR delay EXP-19. "
                    "Both stop at ARA — ensure different platforms."
                ),
            },
            {
                "id": "PASS-21", "type": "Passenger",
                "time": "13:40", "src": "ARA", "dest": "DDU",
                "note": (
                    "Originates at ARA (4 platforms only). Going WEST. "
                    "ARA platform saturation: EXP-19 and EXP-20 also need ARA platforms. "
                    "Assign platform carefully — all 4 can fill up. "
                    "Rain zone: 1.2× travel times on all segments from ARA westward."
                ),
            },
        ],

        "validation_criteria": [
            "FR-24 and RAJ-27 have ≥25 min departure gap from DDU (Freight headway rule)",
            "No HEAD-ON between EXP-19 and EXP-20 on any 2-track segment",
            "ARA Junction: each train assigned distinct platform within 1–4",
            "RAJ-27 not blocked by any lower-priority train (priority_respected ≥ 0.8)",
            "All segment times respect ECR rain 1.2× minimum",
        ],

        "max_steps":      15,
        "baseline_score": 0.67,
    },


    # ─────────────────────────────────────────────────────────────────
    # TASK 3 — HARD
    # "Operation Cyclone Exodus: East Coast Multi-Crisis"
    #
    # Real scenario inspired by Cyclone Phailin (Oct 2013, Category 5):
    # Indian Railways evacuated 980,000 people in 48 hours — the largest
    # cyclone evacuation in Indian history. The operations center at HWH
    # had to simultaneously:
    #   - Keep the premium Rajdhani/Duronto services running (SLA bound)
    #   - Run westbound evacuation trains (VSKP→HWH→NDLS direction)
    #   - Manage freight trains carrying relief supplies (eastbound)
    #   - Handle overnight crossings with day_offset arithmetic
    #   - Cope with 2× travel times in the cyclone-hit SER/ECoR zones
    #
    # ALL 6 physics engine checks fire in this task:
    #   HEAD_ON:        Eastbound NDLS trains vs westbound PNBE/LKO trains
    #   REAR_END:       Freight vs Duronto on HWH→KGP corridor
    #   OVERTAKE:       Shatabdi catches Express on fog-affected NR segment
    #   PLATFORM_CLASH: HWH (3 trains in 90-min window), KGP (4 trains)
    #   TRACK_FULL:     Cyclone zone — 3 eastbound trains on 2-track KGP→BLS
    #   SPEED_VIOLATION: Cyclone enforces 2× minimum — agents who ignore it fail
    #
    # Overnight handling: RAJ-31 and DUR-33 depart ~22:00–23:00 and
    # arrive in the cyclone zone the next day. Stops after midnight
    # must use day_offset=1 with correctly wrapped times.
    # ─────────────────────────────────────────────────────────────────
    "hard_task_3": {
        "difficulty": "hard",

        "scenario_context": (
            "Cyclone Phailin Evacuation Operations — HWH Control Centre. "
            "SER/ECoR zones: Cyclone Alert active, all trains max 30 km/h (2× base times). "
            "NR zone: Dense Fog (1.5× base times). ECR zone: Heavy Rain (1.2× base times). "
            "9 trains must be scheduled — evacuation, premium, and freight — "
            "with conflicting flows, overnight crossings, and platform saturation. "
            "Every failure mode in the physics engine is intentionally triggered "
            "by the naive schedule. The agent must resolve all of them."
        ),

        "description": (
            "Schedule 9 trains across all 7 weather zones during a cyclone crisis. "
            "Conflicts include: HEAD-ON (eastbound vs westbound on 2-track lines), "
            "REAR-END (Freight chasing Duronto out of HWH), "
            "OVERTAKE (Shatabdi catching Express in NR fog), "
            "PLATFORM SATURATION (HWH and KGP), "
            "TRACK FULL (cyclone zone 2-track with 3 eastbound trains), "
            "SPEED VIOLATIONS (agents who ignore cyclone 2× multiplier). "
            "Two trains cross midnight — day_offset=1 required for their eastern stops."
        ),

        "challenge_types": [
            "head_on_multi_segment",    # 3 opposing pairs on different segments
            "rear_end_freight_premium", # FR-13 vs DUR-14 out of HWH
            "overtake_fog_segment",     # SHAT-39 catches EXP-36 in NR zone
            "platform_saturation_hwh",  # 3 trains at HWH in 90-min window
            "platform_saturation_kgp",  # 4 trains at KGP in 2-hour window
            "track_full_cyclone_zone",  # KGP→BLS has 2 tracks, 3 eastbound trains
            "midnight_crossing",        # RAJ-31 and DUR-33: day_offset=1 in cyclone
            "weather_cascade",          # NR fog → ECR rain → SER/ECoR cyclone
            "priority_chain",           # Freight must yield to Duronto, Rajdhani, Shatabdi
            "cyclone_speed_enforcement",# 2× minimum — no shortcuts in ECoR/SER
        ],

        "strategic_hints": [
            # Overnight trains
            "RAJ-31 departs NDLS 22:00. NDLS→HWH takes ~18hrs under fog+rain. Arrives HWH next day ~16:00.",
            "Stations after midnight on RAJ-31/DUR-33 routes need day_offset=1.",
            "DUR-33 departs NDLS 23:00 (1hr after RAJ-31). Both on same NDLS→HWH track. Maintain ≥10min headway at each station.",
            # Cyclone zone
            "In SER/ECoR zones, ALL travel times are 2× base. KGP→BLS 95min becomes 190min. Do not propose faster times.",
            "EXP-12 (HWH→VSKP) enters cyclone at SRC. VSKP journey takes nearly 24hrs under cyclone conditions.",
            # Platform saturation
            "HWH has 23 platforms but EXP-12 (08:00), FR-13 (via ASN, not HWH), DUR-14 (08:30) need HWH platforms.",
            "KGP has 12 platforms. EXP-12, DUR-14, PASS-15, EXP-16 all pass through KGP — stagger by platform AND time.",
            # Head-on
            "EXP-36 (PNBE→NDLS, going WEST) meets RAJ-31/DUR-33 (NDLS→HWH, going EAST) on CNB–PNBE 2-track.",
            "EXP-36 departs PNBE 07:00. RAJ-31 passes PNBE heading east around 13:00 next day — different day_offset, no conflict.",
            "SHAT-39 (LKO→NDLS, going WEST, 150km/h) may catch EXP-36 (PNBE→NDLS, 100km/h) in NR fog zone.",
            # Freight
            "FR-13 (ASN→BHC, Freight) must yield to DUR-14 (HWH→BBS, Duronto). If FR-13 is on KGP→BLS when DUR-14 arrives: REAR-END.",
        ],

        "trains": [
            # ── OVERNIGHT PREMIUM (NDLS → EAST, crossing midnight) ────────
            {
                "id": "RAJ-31", "type": "Rajdhani",
                "time": "22:00", "src": "NDLS", "dest": "BBS",
                "note": (
                    "OVERNIGHT. Departs NDLS 22:00 under Dense Fog (NR). "
                    "Crosses midnight on the CNB–PRYJ segment. "
                    "Stops past midnight: use day_offset=1. "
                    "Enters ECR Rain zone at DDU, then ER Clear, then ECoR Cyclone at BHC. "
                    "In cyclone zone, EVERY segment takes 2×. Do not underestimate journey time. "
                    "Highest priority — all other trains must yield."
                ),
            },
            {
                "id": "DUR-33", "type": "Duronto",
                "time": "23:00", "src": "NDLS", "dest": "HWH",
                "note": (
                    "OVERNIGHT. Departs NDLS 23:00 — 60 min after RAJ-31. "
                    "Same fog corridor. Maintain ≥10 min headway from RAJ-31 at every stop. "
                    "Crosses midnight before GZB — day_offset=1 from GZB onwards. "
                    "Non-stop to HWH (Duronto has very few halts). "
                    "Priority 3 — yields to RAJ-31 but must not be blocked by Freight or Express."
                ),
            },
            # ── EASTBOUND CYCLONE ZONE (HWH → VSKP corridor) ─────────────
            {
                "id": "EXP-12", "type": "Express",
                "time": "08:00", "src": "HWH", "dest": "VSKP",
                "note": (
                    "Enters cyclone zone immediately (HWH is SER boundary). "
                    "HWH→SRC→KGP→BLS→BHC→CTC→BBS→KUR→BAM→VSKP all at 2× time. "
                    "VSKP journey is ~24 hrs under cyclone. Manage day_offset carefully. "
                    "KGP platform competition: DUR-14 follows 30 min later — ensure different platform."
                ),
            },
            {
                "id": "DUR-14", "type": "Duronto",
                "time": "08:30", "src": "HWH", "dest": "BBS",
                "note": (
                    "30 min after EXP-12. Both on HWH→KGP 3-track (safe) then KGP→BLS 2-track. "
                    "Duronto (priority 3) must not be blocked by EXP-12 (priority 5). "
                    "Headway at KGP departure: ensure DUR-14 doesn't rear-end EXP-12. "
                    "Required gap: max(15min Express, 10min Duronto) = 15min."
                ),
            },
            {
                "id": "FR-13", "type": "Freight",
                "time": "07:30", "src": "ASN", "dest": "BHC",
                "note": (
                    "Goods train, priority 7 (LOWEST). Enters cyclone zone at KGP. "
                    "On KGP→BLS 2-track, both EXP-12 and DUR-14 are ahead going east. "
                    "FR-13 must maintain ≥25 min headway from EXP-12/DUR-14. "
                    "If FR-13 is on KGP→BLS when DUR-14 arrives from behind: REAR-END. "
                    "Delay FR-13 ASN departure significantly (try 09:30) to let premium trains clear."
                ),
            },
            {
                "id": "PASS-15", "type": "Passenger",
                "time": "09:00", "src": "SRC", "dest": "KGP",
                "note": (
                    "Short run SRC→KGP in cyclone zone (SER, 2×). "
                    "SRC has only 4 platforms — assign carefully. "
                    "KGP arrival adds to platform saturation (EXP-12, DUR-14, EXP-16 also at KGP). "
                    "OVERTAKE risk: EXP-12 is faster and left HWH 30min earlier — "
                    "PASS-15 starts at SRC (after HWH). Ensure no overtake on SRC→KGP."
                ),
            },
            {
                "id": "EXP-16", "type": "Express",
                "time": "09:15", "src": "KGP", "dest": "CTC",
                "note": (
                    "Departs KGP into deep cyclone zone. "
                    "KGP platform: must be different from EXP-12 and DUR-14 which are also at KGP. "
                    "KGP has 12 platforms — plenty, but stagger dwell times. "
                    "EXP-16 going EAST; EXP-36 going WEST on the same corridor — HEAD-ON risk."
                ),
            },
            # ── WESTBOUND REVERSE FLOW (EVACUATION / RETURN) ─────────────
            {
                "id": "EXP-36", "type": "Express",
                "time": "07:00", "src": "PNBE", "dest": "NDLS",
                "note": (
                    "Going WEST — opposite direction to RAJ-31, DUR-33, EXP-12, DUR-14. "
                    "Check each 2-track segment for head-on: "
                    "  • PNBE→DNR→ARA→BXR→DDU: meets any eastbound train? "
                    "  • RAJ-31/DUR-33 depart NDLS at 22:00/23:00 and reach PNBE ~13:00 next day "
                    "    (day_offset=1) — different day, no conflict. "
                    "  • SHAT-39 (LKO→NDLS) going WEST same direction — check REAR-END/OVERTAKE. "
                    "ECR Rain zone on BXR–DDU segments (1.2×)."
                ),
            },
            {
                "id": "SHAT-39", "type": "Shatabdi",
                "time": "08:00", "src": "LKO", "dest": "NDLS",
                "note": (
                    "Fastest westbound train (150 km/h). Going WEST toward NDLS. "
                    "EXP-36 also going WEST, departed PNBE at 07:00. "
                    "SHAT-39 starts at LKO (different start). They merge on the TDL→GZB segment. "
                    "SHAT-39 may OVERTAKE EXP-36 on GZB→NDLS 4-track — fine (multi-track). "
                    "But on TDL→GZB 2-track: SHAT-39 is behind EXP-36, both going west. "
                    "Headway check required. If SHAT-39 departs TDL too close after EXP-36: REAR-END. "
                    "NR Fog zone on approach to NDLS — respect 1.5× for GZB→NDLS segment."
                ),
            },
        ],

        "validation_criteria": [
            "No HEAD-ON between any eastbound and westbound train on any 2-track segment",
            "RAJ-31 and DUR-33 maintain ≥10 min headway at every shared station",
            "FR-13 maintains ≥25 min headway behind EXP-12 and DUR-14 on KGP→BLS",
            "SHAT-39 does not rear-end EXP-36 on any shared 2-track segment",
            "HWH: EXP-12 and DUR-14 assigned different platforms",
            "KGP: EXP-12, DUR-14, PASS-15, EXP-16 all assigned distinct platforms",
            "RAJ-31/DUR-33 stops after midnight use day_offset=1",
            "All cyclone-zone (SER/ECoR) segment times respect 2× minimum",
            "All fog-zone (NR) segment times respect 1.5× minimum",
            "FR-13 (priority 7) does not block DUR-14 (priority 3) — priority_respected ≥ 0.7",
        ],

        "max_steps":      30,
        "baseline_score": 0.41,
    },
}


# ===================================================================
# 1. DATA MANAGER  —  Persistent storage + learned knowledge base
# ===================================================================
class DataManager:
    def __init__(self, project_root: Path):
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)

        self.network_file   = self.data_dir / "network.json"
        self.timetable_file = self.data_dir / "timetable.json"
        self.state_file     = self.data_dir / "simulation_state.json"
        self.kb_file        = self.data_dir / "knowledge_base.json"

    # ── raw I/O ───────────────────────────────────────────────────────
    def load_json(self, filepath: Path) -> Dict:
        if not filepath.exists():
            return {}
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_json(self, filepath: Path, data: Dict):
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Storage Error on {filepath.name}: {e}")

    # ── network / timetable ───────────────────────────────────────────
    def load_network(self) -> Dict:
        return self.load_json(self.network_file)

    def load_timetable(self) -> List[TrainInfo]:
        data = self.load_json(self.timetable_file)
        return [TrainInfo(**t) for t in data.get("trains", []) if "stops" in t]

    def save_timetable(self, timetable: List[TrainInfo]):
        self.save_json(self.timetable_file, {"trains": [t.model_dump() for t in timetable]})

    def clear_timetable(self):
        self.save_json(self.timetable_file, {"trains": []})

    # ── knowledge base ────────────────────────────────────────────────
    def _load_kb(self) -> Dict:
        kb = self.load_json(self.kb_file)
        kb.setdefault("failures", [])
        kb.setdefault("successes", [])
        kb.setdefault("patterns", {
            "congestion_windows":    {},   # "NDLS-GZB" -> ["06:00", ...]
            "platform_preferences":  {},   # "CNB" -> {"Express": 3, ...}
            "collision_hotspots":    [],   # list of {segment, type, count}
        })
        kb.setdefault("task_scores", {})
        return kb

    def record_failure(self, train_id: str, error_msg: str, track_segment: str = ""):
        import re
        kb = self._load_kb()

        kb["failures"].append({
            "train_id":    train_id,
            "error":       error_msg[:200],
            "track":       track_segment,
        })
        kb["failures"] = kb["failures"][-20:]

        # Learn congestion windows from headway/track-full errors
        if track_segment and any(
            kw in error_msg
            for kw in ("HEADWAY", "TRACK FULL", "REAR-END", "HEAD-ON", "OVERTAKE")
        ):
            windows = kb["patterns"]["congestion_windows"].setdefault(track_segment, [])
            times = re.findall(r"\b\d{2}:\d{2}\b", error_msg)
            if times and times[0] not in windows:
                windows.append(times[0])
                windows[:] = windows[-5:]

        # Track collision hotspots
        if track_segment:
            hotspots = kb["patterns"]["collision_hotspots"]
            existing = next(
                (h for h in hotspots if h["segment"] == track_segment), None
            )
            if existing:
                existing["count"] += 1
            else:
                hotspots.append({"segment": track_segment, "count": 1})
            # Keep only top-10 hotspots
            kb["patterns"]["collision_hotspots"] = sorted(
                hotspots, key=lambda h: h["count"], reverse=True
            )[:10]

        self.save_json(self.kb_file, kb)

    def record_success(self, train: TrainInfo, score: float):
        kb = self._load_kb()

        platform_map = {s.station: s.platform for s in train.stops if s.platform}
        kb["successes"].append({
            "train_id":   train.id,
            "train_type": train.type,
            "route":      [s.station for s in train.stops],
            "platforms":  platform_map,
            "score":      round(score, 3),
        })
        kb["successes"] = kb["successes"][-20:]

        # Update platform preferences — most recent successful choice wins
        for station, plat in platform_map.items():
            prefs = kb["patterns"]["platform_preferences"].setdefault(station, {})
            prefs[train.type] = plat

        self.save_json(self.kb_file, kb)

    def record_task_score(self, task_id: str, score: float):
        kb = self._load_kb()
        kb["task_scores"].setdefault(task_id, []).append(round(score, 3))
        kb["task_scores"][task_id] = kb["task_scores"][task_id][-10:]
        self.save_json(self.kb_file, kb)

    def get_knowledge_base(self) -> Dict:
        return self._load_kb()

    # ── dashboard snapshot ────────────────────────────────────────────
    def save_dashboard_state(self, task_id, step, network, timetable, action, reward):
        self.save_json(self.state_file, {
            "task_id":          task_id,
            "step_count":       step,
            "network_summary":  network,
            "current_timetable": [t.model_dump() for t in timetable],
            "last_action":      action.model_dump() if action else None,
            "last_reward":      reward.model_dump() if reward else None,
        })


# ===================================================================
# 2. PHYSICS ENGINE  —  Best-in-class collision detection & scoring
# ===================================================================
class PhysicsEngine:
    """
    Production-grade Indian Railway physics engine.

    Collision types detected
    ─────────────────────────
    HEAD_ON        Opposite trains on a single-track segment simultaneously.
    REAR_END       Same-direction trains with departure gap < type-specific min headway.
    OVERTAKE       Faster train departs after slower but arrives first on single track
                   (physically impossible — no passing loop available).
    PLATFORM_CLASH Two trains at the same platform with overlapping dwell windows.
    TRACK_FULL     All parallel tracks on a segment occupied by opposing trains.
    SPEED_VIOLATION Proposed travel time < weather-adjusted physics minimum.

    Reward components (weights from config.SCORE_WEIGHTS)
    ──────────────────────────────────────────────────────
    conflict_free      (40%)  Binary — any safety violation sets this to 0
    journey_time       (20%)  Continuous — fraction of segments physically achievable
    platform_valid     (15%)  Binary — any platform over-count sets this to 0
    headway_quality    (12%)  Continuous — avg (actual_gap / required_gap), capped at 1
    priority_respected  (8%)  Continuous — fraction of priority interactions not violated
    dwell_valid         (5%)  Continuous — fraction of intermediate stops with valid dwell
    """

    def __init__(
        self,
        network_summary: Dict,
        train_specs: Dict,
        zone_weather: Dict,
        score_weights: Dict,
    ):
        stations = network_summary.get("stations", [])
        tracks   = network_summary.get("tracks",   [])

        # Station lookups
        self.platforms    = {s["id"]: s.get("platform_count", 5) for s in stations}
        self.is_junction  = {s["id"]: s.get("is_junction", False)  for s in stations}
        self.station_zone = {s["id"]: s.get("zone", "NCR")         for s in stations}

        # Track lookups: (sorted_pair) → {count, distance_km, base_minutes}
        self.tracks: Dict[tuple, Dict] = {}
        for t in tracks:
            pair = tuple(sorted([t["from"], t["to"]]))
            self.tracks[pair] = {
                "count":        t.get("track_count",   2),
                "distance_km":  t.get("distance_km", 100),
                "base_minutes": t.get("travel_minutes", 60),
            }

        self.train_specs  = train_specs
        self.zone_weather = zone_weather
        self.weights      = score_weights

    # ── Helpers ────────────────────────────────────────────────────────
    def _spec(self, train_type: str) -> Dict:
        return self.train_specs.get(train_type, DEFAULT_TRAIN_SPEC)

    def _time_to_mins(self, t: Optional[str]) -> int:
        if not t or ":" not in t:
            return 0
        h, m = map(int, t.split(":"))
        return h * 60 + m

    def _travel_mins(self, s1: StopTime, s2: StopTime) -> int:
        """Actual proposed travel time between two consecutive stops (handles midnight wrap)."""
        dep = self._time_to_mins(s1.departure)
        arr = self._time_to_mins(s2.arrival)
        diff = arr - dep
        return diff + 1440 if diff < 0 else diff  # midnight crossing

    def _dwell_mins(self, stop: StopTime) -> Optional[int]:
        """Actual dwell at a stop. None if arrival or departure is missing."""
        if not stop.arrival or not stop.departure:
            return None
        d = self._time_to_mins(stop.departure) - self._time_to_mins(stop.arrival)
        return d + 1440 if d < 0 else d

    def _is_time_overlap(self, s1: StopTime, s2: StopTime) -> bool:
        """True if the two stop dwell windows overlap (±2 min tolerance)."""
        if s1.day_offset != s2.day_offset:
            return False
        t1_s = self._time_to_mins(s1.arrival  or s1.departure)
        t1_e = self._time_to_mins(s1.departure or s1.arrival)
        t2_s = self._time_to_mins(s2.arrival  or s2.departure)
        t2_e = self._time_to_mins(s2.departure or s2.arrival)
        return max(t1_s, t2_s) <= min(t1_e, t2_e) + 2

    def _min_travel_minutes(self, from_st: str, to_st: str, train_type: str) -> float:
        """
        Minimum physically achievable travel time for a segment.

        Takes the stricter of:
          (a) physics floor:  distance_km / max_speed_kmh × 60
          (b) weather floor:  base_minutes × weather_multiplier × 0.90
              (10% tolerance — accounts for modern rolling stock / favourable gradient)
        """
        pair  = tuple(sorted([from_st, to_st]))
        track = self.tracks.get(pair, {"distance_km": 100, "base_minutes": 60})

        distance_km  = track["distance_km"]
        base_minutes = track["base_minutes"]

        # Weather uses the destination zone (train is entering that zone)
        zone         = self.station_zone.get(to_st, "NCR")
        weather_mult = self.zone_weather.get(zone, {}).get("multiplier", 1.0)

        # Physics absolute floor
        max_speed   = self._spec(train_type)["max_speed_kmh"]
        physics_min = (distance_km / max_speed) * 60.0

        # Weather-adjusted base time (90% = allow minor optimism)
        weather_min = base_minutes * weather_mult * 0.90

        return max(physics_min, weather_min)

    # ── Main evaluation entry point ────────────────────────────────────
    def evaluate(
        self,
        proposal:   ScheduleProposal,
        timetable:  List[TrainInfo],
        train_type: str = "express",
    ) -> Tuple[float, RewardBreakdown, str]:
        """
        Score a schedule proposal against the committed timetable.

        Returns
        ───────
        score      : float in [0.0, 1.0]
        breakdown  : RewardBreakdown with per-component scores
        feedback   : human-readable error summary (empty string = perfect)
        """
        if not proposal or not proposal.stops or len(proposal.stops) < 2:
            return 0.0, RewardBreakdown(), "Invalid or empty proposal."

        errors: List[str] = []
        my_spec = self._spec(train_type)
        stops   = proposal.stops

        # ── Accumulator initialisation ──────────────────────────────────
        has_safety_violation = False   # any → conflict_free = 0

        jt_scores:   List[float] = []  # per-segment journey time scores
        hw_scores:   List[float] = []  # per-interaction headway quality
        prio_checks  = 0
        prio_viol    = 0
        dwell_ok     = 0
        dwell_total  = 0
        platform_ok  = True

        # ══════════════════════════════════════════════════════════════
        # CHECK 1 — PLATFORM CAPACITY & PLATFORM CLASH
        # ══════════════════════════════════════════════════════════════
        for stop in stops:
            max_p = self.platforms.get(stop.station, 1)

            # Over-capacity
            if stop.platform is not None and stop.platform > max_p:
                platform_ok = False
                errors.append(
                    f"PLATFORM ERR: {stop.station} has {max_p} platforms. "
                    f"You assigned platform {stop.platform}."
                )

            # Clash with committed timetable
            for ex in timetable:
                for ex_stop in ex.stops:
                    if (
                        ex_stop.station  == stop.station
                        and ex_stop.platform == stop.platform
                        and stop.platform is not None
                        and self._is_time_overlap(ex_stop, stop)
                    ):
                        has_safety_violation = True
                        errors.append(
                            f"PLATFORM CLASH: {stop.station} Plat-{stop.platform} "
                            f"occupied by {ex.id} at the same time."
                        )

        # ══════════════════════════════════════════════════════════════
        # CHECK 2 — DWELL TIME VALIDATION (intermediate stops)
        # ══════════════════════════════════════════════════════════════
        min_dwell = my_spec["min_dwell"]
        max_dwell = my_spec["max_dwell"]

        for stop in stops[1:-1]:
            dwell = self._dwell_mins(stop)
            if dwell is None:
                continue  # no arrival+departure pair → skip
            dwell_total += 1
            if min_dwell <= dwell <= max_dwell:
                dwell_ok += 1
            elif dwell < min_dwell:
                errors.append(
                    f"DWELL TOO SHORT: {stop.station} dwell={dwell}min "
                    f"(min={min_dwell}min for {train_type})."
                )
            else:
                errors.append(
                    f"DWELL TOO LONG: {stop.station} dwell={dwell}min "
                    f"(max={max_dwell}min for {train_type})."
                )

        # ══════════════════════════════════════════════════════════════
        # CHECK 3 — SEGMENT-LEVEL CHECKS
        # For each consecutive stop pair in the proposal
        # ══════════════════════════════════════════════════════════════
        for i in range(len(stops) - 1):
            s1, s2 = stops[i], stops[i + 1]
            pair       = tuple(sorted([s1.station, s2.station]))
            track_data = self.tracks.get(pair, {"count": 2, "distance_km": 100, "base_minutes": 60})
            max_tracks = track_data["count"]
            my_dep     = self._time_to_mins(s1.departure)
            my_arr     = self._time_to_mins(s2.arrival)
            my_dir     = f"{s1.station}→{s2.station}"

            # ── 3a. Journey Time / Weather / Speed check ───────────────
            actual_travel = self._travel_mins(s1, s2)
            min_required  = self._min_travel_minutes(s1.station, s2.station, train_type)

            if actual_travel > 0:
                seg_jt = min(1.0, actual_travel / min_required) if actual_travel < min_required else 1.0
                jt_scores.append(seg_jt)

                if actual_travel < min_required * 0.80:
                    # Hard violation — >20% faster than physically possible
                    has_safety_violation = True
                    errors.append(
                        f"SPEED VIOLATION: {s1.station}→{s2.station}: "
                        f"proposed {actual_travel}min but physics+weather minimum "
                        f"is {min_required:.0f}min for {train_type} "
                        f"(zone: {self.station_zone.get(s2.station, '?')}, "
                        f"weather: {self.zone_weather.get(self.station_zone.get(s2.station,'NCR'),{}).get('condition','Clear')})."
                    )
                elif actual_travel < min_required:
                    # Soft violation — report but don't zero conflict_free
                    errors.append(
                        f"SPEED WARNING: {s1.station}→{s2.station}: "
                        f"proposed {actual_travel}min is tight "
                        f"(minimum {min_required:.0f}min under current weather)."
                    )

            # ── 3b. Scan committed trains for track interactions ────────
            concurrent_opposing = 0

            for ex in timetable:
                ex_spec = self._spec(ex.type)

                for j in range(len(ex.stops) - 1):
                    es1, es2 = ex.stops[j], ex.stops[j + 1]
                    if pair != tuple(sorted([es1.station, es2.station])):
                        continue

                    ex_dep = self._time_to_mins(es1.departure)
                    ex_arr = self._time_to_mins(es2.arrival)
                    ex_dir = f"{es1.station}→{es2.station}"

                    # Are both trains physically on this segment simultaneously?
                    on_segment = max(my_dep, ex_dep) < min(my_arr, ex_arr)
                    if not on_segment:
                        continue

                    if my_dir != ex_dir:
                        # ── HEAD_ON ──────────────────────────────────────
                        concurrent_opposing += 1
                        if max_tracks == 1:
                            has_safety_violation = True
                            errors.append(
                                f"HEAD-ON CRASH: {s1.station}↔{s2.station} is SINGLE TRACK. "
                                f"{ex.id} ({ex.type}) is heading the opposite direction!"
                            )
                    else:
                        # Same direction — check REAR_END, OVERTAKE, HEADWAY QUALITY, PRIORITY
                        dep_gap      = abs(my_dep - ex_dep)
                        required_hw  = max(my_spec["min_headway"], ex_spec["min_headway"])

                        # ── REAR_END ──────────────────────────────────────
                        if dep_gap < required_hw:
                            has_safety_violation = True
                            errors.append(
                                f"REAR-END RISK: {s1.station}→{s2.station}: "
                                f"only {dep_gap}min gap from {ex.id} ({ex.type}). "
                                f"Required headway: {required_hw}min "
                                f"(max of {my_spec['min_headway']}min/{ex_spec['min_headway']}min)."
                            )

                        # ── OVERTAKE on single track ──────────────────────
                        if max_tracks == 1:
                            i_left_later   = my_dep  > ex_dep
                            i_arrive_first = my_arr  < ex_arr
                            ex_left_later  = ex_dep  > my_dep
                            ex_arrive_first= ex_arr  < my_arr

                            if (i_left_later and i_arrive_first) or (ex_left_later and ex_arrive_first):
                                has_safety_violation = True
                                faster_id = proposal.train_id if i_left_later and i_arrive_first else ex.id
                                slower_id = ex.id if i_left_later and i_arrive_first else proposal.train_id
                                errors.append(
                                    f"OVERTAKE CRASH: {s1.station}→{s2.station} SINGLE TRACK. "
                                    f"{faster_id} would overtake {slower_id} — "
                                    f"no passing loop available."
                                )

                        # ── HEADWAY QUALITY (continuous) ──────────────────
                        hw_scores.append(min(1.0, dep_gap / required_hw))

                        # ── PRIORITY CHECK ────────────────────────────────
                        prio_checks += 1
                        my_prio = my_spec["priority"]
                        ex_prio = ex_spec["priority"]

                        # If our train is lower priority and departs before the higher-priority one
                        # without leaving enough clearance:
                        if my_prio > ex_prio and my_dep < ex_dep and dep_gap < required_hw * 1.5:
                            prio_viol += 1
                            errors.append(
                                f"PRIORITY VIOLATION: {ex.id} ({ex.type}, priority={ex_prio}) "
                                f"blocked by {train_type} (priority={my_prio}) on "
                                f"{s1.station}→{s2.station}. "
                                f"Give the higher-priority train at least {required_hw * 1.5:.0f}min clearance."
                            )

            # ── TRACK FULL ────────────────────────────────────────────────
            if concurrent_opposing >= max_tracks:
                has_safety_violation = True
                errors.append(
                    f"TRACK FULL: {s1.station}↔{s2.station} "
                    f"({max_tracks} track{'s' if max_tracks > 1 else ''}) — "
                    f"all lines occupied by opposing trains."
                )

        # ══════════════════════════════════════════════════════════════
        # COMPUTE COMPONENT SCORES
        # ══════════════════════════════════════════════════════════════
        bdown = RewardBreakdown(
            conflict_free=      0.0 if has_safety_violation else 1.0,
            journey_time=       sum(jt_scores) / len(jt_scores) if jt_scores else 1.0,
            platform_valid=     0.0 if not platform_ok else 1.0,
            headway_quality=    sum(hw_scores) / len(hw_scores) if hw_scores else 1.0,
            priority_respected= max(0.0, 1.0 - prio_viol / max(prio_checks, 1)),
            dwell_valid=        dwell_ok / dwell_total if dwell_total > 0 else 1.0,
        )

        # Weighted final score (weights come from config — not hardcoded here)
        score = sum(
            getattr(bdown, component) * weight
            for component, weight in self.weights.items()
        )

        return round(score, 3), bdown, " | ".join(errors) or "Perfect Schedule."


# ===================================================================
# 3. DISRUPTION ENGINE  —  Stochastic real-world chaos
# ===================================================================
class DisruptionEngine:
    """Injects random signal failures and track blockages to force agent adaptability."""

    EVENTS = [
        "SIGNAL FAILURE: New arrivals on Main Line delayed by {delay}min.",
        "TRACK BLOCKAGE: Engineering work — single-line working in effect for {delay}min.",
        "LOCO FAILURE: Preceding train stalled — expect {delay}min clearance delay.",
        "LEVEL CROSSING INCIDENT: Traffic jam at gate — {delay}min block.",
    ]

    @classmethod
    def generate(cls, probability: float = 0.05) -> str:
        if random.random() < probability:
            delay = random.randint(15, 60)
            return random.choice(cls.EVENTS).format(delay=delay)
        return ""


# ===================================================================
# 4. GRADER  —  Task-level 0.0–1.0 score
# ===================================================================
def grade_task(
    task_id:          str,
    trains_scheduled: int,
    trains_total:     int,
    avg_score:        float,
) -> float:
    """
    Grade a completed task.
      50% completion rate  +  50% average physics quality
    Returns a value in [0.0, 1.0].
    """
    if trains_total == 0:
        return 0.0
    completion = trains_scheduled / trains_total
    return round(0.5 * completion + 0.5 * avg_score, 3)


# ===================================================================
# 5. ORCHESTRATOR  —  Main OpenEnv interface
# ===================================================================
class TrainSchedulingEnv(Environment):
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent

        self.db              = DataManager(project_root)
        self.network_summary = self.db.load_network()
        self.physics         = PhysicsEngine(
            self.network_summary, TRAIN_SPECS, ZONE_WEATHER, SCORE_WEIGHTS
        )
        self.chaos = DisruptionEngine()

        # Task state
        self.active_task_id:   str       = "default"
        self.task_trains:      List[Dict] = []
        self.trains_scheduled: int        = 0
        self.scores_this_task: List[float]= []

        # Episode state
        self.task_id   = str(uuid.uuid4())[:8]
        self.step_count= 0
        self.max_steps = 15
        self.episode_id= f"train_sched_{self.task_id}"
        self.timetable: List[TrainInfo] = []
        self.current_request = self._generate_mock_request()

    # ── OpenEnv required ──────────────────────────────────────────────
    @property
    def state(self) -> State:
        return State(episode_id=self.episode_id, step_count=self.step_count)

    def reset(self) -> TrainObservation:
        self.step_count        = 0
        self.task_id           = str(uuid.uuid4())[:8]
        self.trains_scheduled  = 0
        self.scores_this_task  = []
        self.timetable         = []
        self.db.clear_timetable()
        self.current_request   = self._generate_mock_request()
        return self._get_obs()

    def step(self, action: TrainAction) -> TrainObservation:
        self.step_count += 1
        disruption = self.chaos.generate()

        # Determine train type: prefer explicit field, then infer from ID
        proposal = action.schedule_proposal
        if proposal:
            train_type = proposal.train_type or infer_train_type(proposal.train_id)
        else:
            train_type = "express"

        # Physics evaluation
        score, breakdown, feedback = self.physics.evaluate(
            proposal, self.timetable, train_type
        )
        if disruption:
            feedback = f"{disruption} | {feedback}"

        # Reward IS the physics score (0.0–1.0, no binary transform)
        rl_reward = score

        # Knowledge base updates
        if proposal:
            train_id = proposal.train_id
            if score >= 0.6:
                new_train = TrainInfo(
                    id=train_id,
                    name=f"Service {train_id}",
                    type=train_type,
                    days=["Daily"],
                    stops=proposal.stops,
                )
                self.timetable.append(new_train)
                self.db.save_timetable(self.timetable)
                self.db.record_success(new_train, score)
                self.trains_scheduled += 1
                self.scores_this_task.append(score)
            else:
                seg = (
                    f"{proposal.stops[0].station}-{proposal.stops[1].station}"
                    if len(proposal.stops) >= 2 else ""
                )
                self.db.record_failure(train_id, feedback, seg)

        # Episode boundary
        all_done = (
            len(self.task_trains) > 0
            and self.trains_scheduled >= len(self.task_trains)
        )
        done = self.step_count >= self.max_steps or all_done

        if done and self.scores_this_task:
            avg = sum(self.scores_this_task) / len(self.scores_this_task)
            final = grade_task(
                self.active_task_id,
                self.trains_scheduled,
                max(len(self.task_trains), 1),
                avg,
            )
            self.db.record_task_score(self.active_task_id, final)

        # Dashboard snapshot
        reward_obj = TrainReward(score=score, breakdown=breakdown, feedback=feedback, done=done)
        self.db.save_dashboard_state(
            self.task_id, self.step_count,
            self.network_summary, self.timetable, action, reward_obj,
        )

        return self._get_obs(reward=rl_reward, done=done, server_msg=feedback)

    # ── Task loading ───────────────────────────────────────────────────
    def load_task(self, task_id: str):
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY)}"
            )
        task = TASK_REGISTRY[task_id]
        self.active_task_id = task_id
        self.task_trains    = task["trains"]
        self.max_steps      = task["max_steps"]

    # ── Observation builder ────────────────────────────────────────────
    def _get_obs(
        self,
        reward: float = 0.0,
        done:   bool  = False,
        server_msg: str = "",
    ) -> TrainObservation:
        kb = self.db.get_knowledge_base()

        # Compact, agent-useful KB summary — only recent + pattern data
        kb_summary = {
            "recent_failures":      kb["failures"][-5:],
            "recent_successes":     [
                {"train_id": s["train_id"], "type": s["train_type"],
                 "platforms": s["platforms"], "score": s["score"]}
                for s in kb["successes"][-5:]
            ],
            "platform_preferences": kb["patterns"].get("platform_preferences", {}),
            "congestion_windows":   kb["patterns"].get("congestion_windows", {}),
            "collision_hotspots":   kb["patterns"].get("collision_hotspots", [])[:5],
        }

        return TrainObservation(
            task_id=self.task_id,
            goal=(
                f"[{self.active_task_id.upper()}] "
                "Schedule each train safely. "
                "Physics engine enforces: headway rules, head-on/rear-end/overtake prevention, "
                "platform capacity, weather-adjusted travel times, dwell limits. "
                "Use metadata.knowledge_base to avoid repeating known mistakes."
            ),
            network_summary=self.network_summary,
            timetable=self.timetable,
            new_train_request=self.current_request,
            step_count=self.step_count,
            max_steps=self.max_steps,
            reward=reward,
            done=done,
            server_feedback=server_msg,
            metadata={"knowledge_base": kb_summary},
        )

    def _generate_mock_request(self) -> TrainRequest:
        hour = random.randint(7, 10)
        return TrainRequest(
            origin="NDLS", destination="DDU",
            stops=["CNB", "PRYJ"],
            train_type="superfast",
            days=["Daily"],
            preferred_departure=f"{hour:02d}:00",
        )
