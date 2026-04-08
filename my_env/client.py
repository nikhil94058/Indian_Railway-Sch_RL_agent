from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Ensure this says .model (singular)
from .models import TrainAction, TrainObservation

class MyEnv(EnvClient[TrainAction, TrainObservation, State]):
    def _step_payload(self, action: TrainAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TrainObservation]:
        """Parses the server's JSON response back into TrainObservation."""
        
        # Safely extract the observation dictionary
        obs_data = payload.get("observation", {})
        
        observation = TrainObservation(
            task_id=obs_data.get("task_id", "unknown"),
            goal=obs_data.get("goal", ""),
            network_summary=obs_data.get("network_summary", {}),
            timetable=obs_data.get("timetable", []),
            new_train_request=obs_data.get("new_train_request"),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
            server_feedback=obs_data.get("server_feedback", "")
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )