"""
오케스트레이터 에이전트 노드.
담당:
  1. 크로스베이 요청 — LLM으로 실행 타당성 판단 후 전체 베이 제어
  2. 충돌/불확실 고위험 — 알림 발송 + 관리자 에스컬레이션
  3. 보고서 이상 패턴 — 관리자 알림 발송
"""
import asyncio
import json
from openai import OpenAI

from llm_module.state import FacilityState
from llm_module.alert_manager import handle as alert_handle, _send_push
from llm_module.temperature_controller import apply_customer_pref
from llm_module.vlm_analyzer import AnalysisResult, DetectionType, Severity

client = OpenAI()

_CROSS_BAY_SYSTEM = """
당신은 무인 스크린골프장 통합 오케스트레이터 AI입니다.
고객이 시설 전체(전체 타석)에 영향을 주는 환경 제어를 요청했습니다.

다음 기준으로 요청을 판단하세요:
- 온도/팬/조명은 전체 타석에 동일하게 적용해도 일반적으로 무방합니다.
- 단, 요청값이 시설 운영 범위(18~30도)를 벗어나면 거부하세요.
- 게임 진행에 방해가 될 수 있는 갑작스러운 변경이면 단계적 적용을 제안하세요.

반드시 JSON으로만 응답하세요:
{
  "approved": true | false,
  "reason": "판단 근거 한 문장",
  "adjusted_value": <조정된 값 또는 원래 값, 숫자>,
  "announcement": "다른 타석 고객에게 안내할 짧은 문장 (승인 시)"
}
"""


async def _judge_cross_bay(action: dict, bay_id: str) -> dict:
    """크로스베이 요청의 타당성을 LLM으로 판단."""
    user_content = (
        f"요청 타석: {bay_id}\n"
        f"요청 내용: {json.dumps(action, ensure_ascii=False)}"
    )
    raw = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _CROSS_BAY_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=150,
    )
    return json.loads(raw.choices[0].message.content)


async def orchestrator_node(state: FacilityState) -> dict:
    print(f"[AGENT: orchestrator] bay={state['bay_id']} conflict={state.get('conflict_detected')} cross_bay={bool(state.get('cross_bay_request'))}")
    bay_id = state["bay_id"]
    all_bay_ids = state.get("all_bay_ids") or []
    decisions: list[str] = []

    # 1. 크로스베이 요청 — LLM 판단 후 실행
    cross_bay = state.get("cross_bay_request")
    if cross_bay and cross_bay.get("action"):
        action = cross_bay["action"]
        judgment = await _judge_cross_bay(action, bay_id)
        print(f"[ORCHESTRATOR] 크로스베이 판단: approved={judgment['approved']} / {judgment['reason']}")

        if judgment["approved"]:
            value = judgment.get("adjusted_value")
            for bid in all_bay_ids:
                if action.get("type") == "temperature" and value is not None:
                    await apply_customer_pref(bid, float(value))
            decisions.append(
                f"크로스베이 {action.get('type')} 승인 및 적용 (값={value}) / {judgment['reason']}"
            )
            if judgment.get("announcement"):
                print(f"[ORCHESTRATOR] 안내: {judgment['announcement']}")
        else:
            decisions.append(f"크로스베이 요청 거부: {judgment['reason']}")

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
