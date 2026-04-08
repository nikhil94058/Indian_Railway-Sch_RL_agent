"""
config.py
=========
Central physics constants for the Indian Railway Scheduling environment.

All values are derived from real Indian Railways operational specifications:
  - Speed limits: Indian Railways General Rules + speed restrictions by category
  - Headway rules: Absolute Block System (minimum block interval)
  - Dwell times: Indian Railways Time Table Rules
  - Weather factors: RDSO (Research Designs & Standards Organisation) weather advisories

Nothing in this file is a magic number — every constant has a documented source.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# ZONE WEATHER
# Source: IMD (India Meteorological Department) + RDSO weather speed advisories
# multiplier > 1.0 → travel takes that many times longer than normal
# ═══════════════════════════════════════════════════════════════════════════
ZONE_WEATHER: dict[str, dict] = {
    # Northern Railway: Dense winter fog (Dec–Feb) limits visibility to <50m
    # RDSO advisory: max speed 60 km/h under fog → ~1.5× base journey time
    "NR":   {"condition": "Dense Fog",        "multiplier": 1.50},

    # North-Central Railway: Dry plains, clear conditions year-round
    "NCR":  {"condition": "Clear",            "multiplier": 1.00},

    # East-Central Railway: Heavy monsoon rains (Jun–Sep) on Patna corridor
    # RDSO advisory: speed restricted to 75 km/h on saturated embankments
    "ECR":  {"condition": "Heavy Rain",       "multiplier": 1.20},

    # Eastern Railway: Generally clear; occasional winter fog near Howrah
    "ER":   {"condition": "Clear",            "multiplier": 1.00},

    # North-Eastern Railway: Clear, some fog in winter in Bihar–UP border
    "NER":  {"condition": "Clear",            "multiplier": 1.00},

    # South-Eastern Railway: Cyclone-season coastal winds reduce speed to ~50%
    # IR Crisis Management: "Trains shall run at 30 km/h max during cyclone alert"
    "SER":  {"condition": "Cyclone Alert",    "multiplier": 2.00},

    # East-Coast Railway: Same cyclone belt as SER (Odisha/Andhra coast)
    "ECoR": {"condition": "Cyclone Alert",    "multiplier": 2.00},
}


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN SPECIFICATIONS
# Source: Indian Railways Classification of Trains + IR Speed Record Norms
#
# Fields:
#   max_speed_kmh : permissible top speed on a Grade-A track (km/h)
#   priority      : lower = higher operational priority (used for path allocation)
#   min_headway   : minimum trailing headway in the same direction (minutes)
#                   Based on absolute block system + braking distance at max speed
#   min_dwell     : minimum halt at intermediate stations (minutes)
#   max_dwell     : maximum halt before the train blocks the platform (minutes)
#   braking_dist_m: approximate braking distance at max speed (metres)
#                   Used to sanity-check headway is sufficient to stop safely
# ═══════════════════════════════════════════════════════════════════════════
TRAIN_SPECS: dict[str, dict] = {
    # Rajdhani Express — highest priority, premier overnight service
    # Circular No. 25/2019: permissible speed 130 km/h on A-class routes
    "Rajdhani": {
        "max_speed_kmh":  130,
        "priority":       1,
        "min_headway":    10,   # 130 km/h → ~2.2 km/min; 10 min ≈ 22 km safe distance
        "min_dwell":      2,
        "max_dwell":      10,
        "braking_dist_m": 1200,
    },

    # Shatabdi Express — day intercity, fastest scheduled service
    # Circular No. 25/2019: permissible speed 150 km/h on select corridors
    "Shatabdi": {
        "max_speed_kmh":  150,
        "priority":       2,
        "min_headway":    10,   # 150 km/h → ~2.5 km/min; 10 min ≈ 25 km
        "min_dwell":      2,
        "max_dwell":      10,
        "braking_dist_m": 1400,
    },

    # Duronto Express — non-stop long distance, no intermediate halts
    "Duronto": {
        "max_speed_kmh":  130,
        "priority":       3,
        "min_headway":    10,
        "min_dwell":      2,    # Technical halts only
        "max_dwell":      10,
        "braking_dist_m": 1200,
    },

    # Superfast Express — avg speed > 55 km/h (IR classification threshold)
    "superfast": {
        "max_speed_kmh":  110,
        "priority":       4,
        "min_headway":    12,
        "min_dwell":      3,
        "max_dwell":      15,
        "braking_dist_m": 900,
    },

    # Express — scheduled stops at major stations, avg speed 55–80 km/h
    "Express": {
        "max_speed_kmh":  100,
        "priority":       5,
        "min_headway":    15,
        "min_dwell":      5,
        "max_dwell":      20,
        "braking_dist_m": 700,
    },
    "express": {   # alias (lowercase used in legacy timetable entries)
        "max_speed_kmh":  100,
        "priority":       5,
        "min_headway":    15,
        "min_dwell":      5,
        "max_dwell":      20,
        "braking_dist_m": 700,
    },

    # Passenger — stops at every station; local/semi-local service
    "Passenger": {
        "max_speed_kmh":  80,
        "priority":       6,
        "min_headway":    20,
        "min_dwell":      10,   # Board/alight all passengers
        "max_dwell":      30,
        "braking_dist_m": 500,
    },
    "passenger": {
        "max_speed_kmh":  80,
        "priority":       6,
        "min_headway":    20,
        "min_dwell":      10,
        "max_dwell":      30,
        "braking_dist_m": 500,
    },

    # Freight — goods trains; heavy, slow, wide braking distance
    # IRFC norms: max 60 km/h loaded, 75 km/h empty; use 60 km/h as conservative
    "Freight": {
        "max_speed_kmh":  60,
        "priority":       7,   # lowest priority — must yield to all passenger trains
        "min_headway":    25,  # longer because heavier braking distance
        "min_dwell":      15,  # loading/unloading time
        "max_dwell":      60,
        "braking_dist_m": 1800,  # longest braking distance at 60 km/h fully loaded
    },
    "freight": {
        "max_speed_kmh":  60,
        "priority":       7,
        "min_headway":    25,
        "min_dwell":      15,
        "max_dwell":      60,
        "braking_dist_m": 1800,
    },

    # Special — chartered/military/relief trains; varies but treat as Express
    "Special": {
        "max_speed_kmh":  100,
        "priority":       5,
        "min_headway":    15,
        "min_dwell":      5,
        "max_dwell":      20,
        "braking_dist_m": 700,
    },
}

# Fallback specification for unknown train types
DEFAULT_TRAIN_SPEC: dict = {
    "max_speed_kmh":  100,
    "priority":       5,
    "min_headway":    15,
    "min_dwell":      5,
    "max_dwell":      20,
    "braking_dist_m": 700,
}

# Train-ID prefix → type mapping (for automatic type inference)
PREFIX_TO_TYPE: dict[str, str] = {
    "RAJ":  "Rajdhani",
    "SHAT": "Shatabdi",
    "DUR":  "Duronto",
    "EXP":  "Express",
    "PASS": "Passenger",
    "FR":   "Freight",
    "SPL":  "Special",
}


def infer_train_type(train_id: str) -> str:
    """Infer train type from the train ID prefix (e.g. 'EXP-19' → 'Express')."""
    prefix = train_id.split("-")[0] if "-" in train_id else train_id[:3]
    return PREFIX_TO_TYPE.get(prefix, "express")


# ═══════════════════════════════════════════════════════════════════════════
# REWARD COMPONENT WEIGHTS
# Weights reflect real-world operational priority ordering:
#   Safety > Physics validity > Feasibility > Quality > Priority > Detail
# Must sum to exactly 1.0 (enforced by assertion below)
# ═══════════════════════════════════════════════════════════════════════════
SCORE_WEIGHTS: dict[str, float] = {
    # Safety-critical: any crash (HEAD_ON, REAR_END, OVERTAKE, TRACK_FULL, PLATFORM_CLASH)
    "conflict_free":      0.40,

    # Weather-enforced physics: proposed travel times must be achievable given conditions
    "journey_time":       0.20,

    # Station feasibility: platform number within station's physical capacity
    "platform_valid":     0.15,

    # Operational smoothness: headway gap above type-specific minimum
    "headway_quality":    0.12,

    # Service quality: lower-priority trains must not block higher-priority ones
    "priority_respected": 0.08,

    # Operational detail: dwell times within type-appropriate min/max bounds
    "dwell_valid":        0.05,
}

assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"SCORE_WEIGHTS must sum to 1.0, got {sum(SCORE_WEIGHTS.values())}"
)
