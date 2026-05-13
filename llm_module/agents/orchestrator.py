"""
오케스트레이터 에이전트 노드.
담당:
  1. 크로스베이 요청 — 시설 전체 환경 제어 적용
  2. 충돌/불확실 고위험 — 알림 발송 + 관리자 에스컬레이션
  3. 보고서 이상 패턴 — 관리자 알림 발송
"""
from llm_module.state import FacilityState
from llm_module.alert_manager import handle as alert_handle, _send_push
from llm_module.temperature_controller import apply_customer_pref
from llm_module.vlm_analyzer import AnalysisResult, DetectionType, Severity


async def orchestrator_node(state: FacilityState) -> dict:
    bay_id = state["bay_id"]
    all_bay_ids = state.get("all_bay_ids") or []
    decisions: list[str] = []

    # 1. 크로스베이 요청 처리
    cross_bay = state.get("cross_bay_request")
    if cross_bay and cross_bay.get("action"):
        action = cross_bay["action"]
        for bid in all_bay_ids:
            if action.get("type") == "temperature" and action.get("value"):
                await apply_customer_pref(bid, float(action["value"]))
        decisions.append(f"전체 베이 {action.get('type')} 적용")
        print(f"[ORCHESTRATOR] 크로스베이 {action.get('type')} → {all_bay_ids}")

    # 2. 충돌/불확실 고위험 처리
    if state.get("conflict_detected"):
        result_dict = state.get("analysis_result") or {}
        if result_dict:
            result = AnalysisResult(
                detection_type=DetectionType(result_dict["detection_type"]),
                detected=True,
                confidence=result_dict["confidence"],
                severity=Severity(result_dict["severity"]),
                evidence=result_dict["evidence"],
                action_required=result_dict["action_required"],
            )
            await alert_handle(result, bay_id)
            decisions.append(f"충돌 감지: {result_dict['detection_type']} 관리자 알림 발송")
            print(f"[ORCHESTRATOR] 충돌 감지 → 관리자 알림: {bay_id}")

    # 3. 보고서 이상 패턴 처리
    if state.get("anomaly_detected") and state.get("report_text"):
        preview = (state["report_text"] or "")[:200]
        await _send_push("일일 운영 이상 감지", preview)
        decisions.append("보고서 이상 패턴: 관리자 알림 발송")
        print("[ORCHESTRATOR] 보고서 이상 → 관리자 푸시 발송")

    return {
        "orchestrator_decision": "; ".join(decisions) if decisions else "처리 완료",
        "escalate_to_human": False,
    }
