"""코칭 에이전트 노드. 기존 coaching_engine.posture_feedback() 래핑."""
from llm_module.state import FacilityState
from llm_module.coaching_engine import posture_feedback


async def coaching_node(state: FacilityState) -> dict:
    pose_keypoints = state.get("pose_keypoints") or {}
    bay_id = state["bay_id"]
    audio_path = await posture_feedback(pose_keypoints, bay_id)
    return {"coaching_audio_path": audio_path}
