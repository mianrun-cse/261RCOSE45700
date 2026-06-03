"""
실행(actuator) 노드.
결정 노드들이 state["pending_actions"]에 적어 둔 실행 의도를 일괄 수행한다.
부수효과(알림 발송 / 환경 제어 / TTS 생성 / 관리자 푸시)는 오직 이 노드에서만 발생한다.

pending_actions 항목 형태 (kind별 dict):
  {"kind": "alert", "result": <AnalysisResult 직렬화 dict>}
  {"kind": "temperature", "temperature": float, "humidity": float,
                          "reason": str, "target_temp_delta": float}
  {"kind": "temperature_all", "value": float}
  {"kind": "push", "title": str, "body": str}
  {"kind": "tts", "text": str, "filename": str}
"""
from llm_module.state import FacilityState
from llm_module.alert_manager import handle as alert_handle, _send_push
from llm_module.temperature_controller import handle as temp_handle, apply_customer_pref
from llm_module.tts import _tts
from llm_module.vlm_analyzer import AnalysisResult, DetectionType, Severity


def _to_analysis_result(d: dict) -> AnalysisResult:
    """직렬화된 dict를 AnalysisResult로 복원."""
    return AnalysisResult(
        detection_type=DetectionType(d["detection_type"]),
        detected=d.get("detected", True),
        confidence=d["confidence"],
        severity=Severity(d["severity"]),
        evidence=d["evidence"],
        action_required=d["action_required"],
    )


async def actuator_node(state: FacilityState) -> dict:
    actions = state.get("pending_actions") or []
    zone_id = state["zone_id"]
    all_zone_ids = state.get("all_zone_ids") or [zone_id]
    print(f"[AGENT: actuator] zone={zone_id} actions={len(actions)}")

    bot_response = state.get("bot_response")
    updates: dict = {}

    for action in actions:
        kind = action.get("kind")

        if kind == "alert":
            result = _to_analysis_result(action["result"])
            await alert_handle(result, zone_id)

        elif kind == "temperature":
            await temp_handle(
                zone_id=zone_id,
                temperature=action.get("temperature", 25.0),
                humidity=action.get("humidity", 60.0),
                reason=action.get("reason", ""),
                target_temp_delta=action.get("target_temp_delta", -1.0),
            )

        elif kind == "temperature_all":
            value = action.get("value")
            if value is not None:
                for zid in all_zone_ids:
                    await apply_customer_pref(zid, float(value))

        elif kind == "push":
            await _send_push(action.get("title", ""), action.get("body", ""))

        elif kind == "tts":
            path = await _tts(action["text"], action["filename"])
            if bot_response is not None:
                bot_response = {**bot_response, "audio_path": str(path)}
                updates["bot_response"] = bot_response

        else:
            print(f"[ACTUATOR] 알 수 없는 action kind: {kind}")

    return updates
