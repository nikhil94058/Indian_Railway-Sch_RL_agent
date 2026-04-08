"""
models.py
=========
Pydantic models defining the Observation, Action, and Reward spaces
for the Indian Railway Scheduling RL Environment.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ══════════════════════════════════════════════════════════════════════
# ── OBSERVATION SPACE ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class StopTime(BaseModel):
    """Details for a specific stop in a train's journey."""
    station: str
    arrival:    Optional[str] = Field(None, description="Arrival time 'HH:MM' (null for origin)")
    departure:  Optional[str] = Field(None, description="Departure time 'HH:MM' (null for destination)")
    platform:   Optional[int] = Field(None, description="Assigned platform number (1-indexed)")
    day_offset: int            = Field(0,    description="0 = departure day, 1 = next day, etc.")


class TrainInfo(BaseModel):
    """Full schedule and metadata for a single committed train."""
    id:   str
    name: str
    type: str  # Rajdhani | Shatabdi | Duronto | superfast | Express | Passenger | Freight | Special
    days: List[str] = Field(..., description="Operating days, e.g. ['Mon', 'Wed'] or ['Daily']")
    stops: List[StopTime]


class Conflict(BaseModel):
    """Description of a detected scheduling conflict."""
    train_a:       str
    train_b:       str
    location:      str
    conflict_type: str  # HEAD_ON | REAR_END | OVERTAKE | PLATFORM_CLASH | TRACK_FULL
    time_window:   str  = Field(..., description="Time range, e.g. '08:00–08:15'")


class TrainRequest(BaseModel):
    """The scheduling task assigned to the agent for this step."""
    origin:             str
    destination:        str
    stops:              List[str] = Field(..., description="Intermediate station IDs to serve")
    train_type:         str
    preferred_departure: Optional[str] = None
    days:               List[str]


class TrainObservation(BaseModel):
    """
    Full state of the environment returned to the agent at each step.

    Key fields for the agent:
      - timetable: all trains committed so far (use for conflict avoidance)
      - new_train_request: the train the agent must schedule next
      - metadata.knowledge_base: learned patterns (platform prefs, congestion windows)
      - server_feedback: physics engine error message from the last action
    """
    task_id:          str
    goal:             str
    network_summary:  Dict[str, Any] = Field(default_factory=dict)
    timetable:        List[TrainInfo] = Field(default_factory=list)
    new_train_request: Optional[TrainRequest] = None
    step_count:       int
    max_steps:        int
    reward:           float = 0.0
    done:             bool  = False
    metadata:         Dict[str, Any] = Field(default_factory=dict)
    server_feedback:  str = ""


# ══════════════════════════════════════════════════════════════════════
# ── ACTION SPACE ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

class ConflictFlag(BaseModel):
    """Structured conflict report the agent may optionally include."""
    train_a:       str
    train_b:       str
    location:      str
    conflict_type: str
    description:   str


class ScheduleProposal(BaseModel):
    """A complete proposed schedule for one train service."""
    train_id:   str
    train_type: Optional[str] = Field(
        None,
        description=(
            "Train category (Rajdhani/Shatabdi/Duronto/Express/Passenger/Freight/Special). "
            "Used by the physics engine for type-specific headway and dwell validation. "
            "If omitted, the engine infers it from the train_id prefix."
        ),
    )
    stops: List[StopTime]


class TrainAction(BaseModel):
    """
    The action sent by the agent to the environment each step.
    Reasoning is mandatory — it enables explainability scoring.
    """
    action_type: str = Field(
        ...,
        description="One of: generate_schedule | detect_conflicts | suggest_slot | noop",
    )
    conflict_flags:    Optional[List[ConflictFlag]] = None
    schedule_proposal: Optional[ScheduleProposal]  = None
    reasoning: str = Field(
        ...,
        description=(
            "Chain-of-thought explaining the scheduling decision: "
            "which conflicts were identified, what delays were applied, "
            "which knowledge-base patterns were used."
        ),
    )


# ══════════════════════════════════════════════════════════════════════
# ── REWARD SPACE ──────────────────────────────════════════════════════
# ══════════════════════════════════════════════════════════════════════

class RewardBreakdown(BaseModel):
    """
    Six-component reward breakdown from the physics engine.

    All components are in [0.0, 1.0].
    Final score = weighted sum (weights defined in data/config.py SCORE_WEIGHTS).

    Component   Weight  What it measures
    ─────────── ─────── ────────────────────────────────────────────────────
    conflict_free   40% No safety violations:
                          HEAD_ON, REAR_END, OVERTAKE, PLATFORM_CLASH, TRACK_FULL
    journey_time    20% Travel times physically achievable under current weather
                          (weather zone multiplier + train max-speed floor)
    platform_valid  15% Platform number ≤ station's physical platform count
    headway_quality 12% Headway gap as fraction of type-specific minimum
                          (continuous — rewards wider safety margins)
    priority_respected 8% Lower-priority trains did not block higher-priority ones
    dwell_valid      5% Intermediate stop dwell within [min_dwell, max_dwell]
                          for the specific train type
    """
    conflict_free:       float = Field(0.0, ge=0.0, le=1.0)
    journey_time:        float = Field(0.0, ge=0.0, le=1.0)
    platform_valid:      float = Field(0.0, ge=0.0, le=1.0)
    headway_quality:     float = Field(0.0, ge=0.0, le=1.0)
    priority_respected:  float = Field(0.0, ge=0.0, le=1.0)
    dwell_valid:         float = Field(0.0, ge=0.0, le=1.0)


class TrainReward(BaseModel):
    """Final reward object returned by the environment after each step."""
    score:     float = Field(..., ge=0.0, le=1.0, description="Weighted total score (0.0–1.0)")
    breakdown: RewardBreakdown
    feedback:  str   = Field(..., description="Physics engine natural-language error summary")
    done:      bool
