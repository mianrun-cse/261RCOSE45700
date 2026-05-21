"""
오케스트레이터 에이전트 노드 (결정 전용).
담당:
  1. 크로스존 요청 — LLM으로 실행 타당성 판단 (실행은 actuator)
  2. 충돌/불확실 고위험 — 알림 발송 의도 발행
  3. 보고서 이상 패턴 — 관리자 푸시 의도 발행
부수효과는 직접 실행하지 않고 pending_actions로만 발행한다.
"""
import asyncio
import json
from openai import OpenAI

from llm_module.state import FacilityState

client = OpenAI()

_CROSS_ZONE_SYSTEM = """
당신은 무인 매장(무인 카페·편의점 등) 통합 오케스트레이터 AI입니다.
고객이 매장 전체(전체 구역)에 영향을 주는 환경 제어를 요청했습니다.

다음 기준으로 요청을 판단하세요:
- 온도/팬/조명은 전체 구역에 동일하게 적용해도 일반적으로 무방합니다.
- 단, 요청값이 시설 운영 범위(18~30도)를 벗어나면 거부하세요.
- 다른 고객 이용에 방해가 될 수 있는 갑작스러운 변경이면 단계적 적용을 제안하세요.

반드시 JSON으로만 응답하세요:
{
  "approved": true | false,
  "reason": "판단 근거 한 문장",
  "adjusted_value": <조정된 값 또는 원래 값, 숫자>,
  "announcement": "다른 구역 고객에게 안내할 짧은 문장 (승인 시)"
}
"""


async def _judge_cross_zone(action: dict, zone_id: str) -> dict:
    """크로스존 요청의 타당성을 LLM으로 판단."""
    user_content = (
        f"요청 구역: {zone_id}\n"
        f"요청 내용: {json.dumps(action, ensure_ascii=False)}"
    )
    raw = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _CROSS_ZONE_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=150,
    )
    return json.loads(raw.choices[0].message.content)


async def orchestrator_node(state: FacilityState) -> dict:
    print(f"[AGENT: orchestrator] zone={state['zone_id']} conflict={state.get('conflict_detected')} cross_zone={bool(state.get('cross_zone_request'))}")
    zone_id = state["zone_id"]
    decisions: list[str] = []
    actions: list[dict] = []

    # 1. 크로스존 요청 — LLM 판단 후 실행 의도 발행
    cross_zone = state.get("cross_zone_request")
    if cross_zone and cross_zone.get("action"):
        action = cross_zone["action"]
        judgment = await _judge_cross_zone(action, zone_id)
        approved = bool(judgment.get("approved", False))
        reason = judgment.get("reason", "사유 미상")
        print(f"[ORCHESTRATOR] 크로스존 판단: approved={approved} / {reason}")

        if approved:
            value = judgment.get("adjusted_value")
            if action.get("type") == "temperature" and value is not None:
                try:
                    actions.append({"kind": "temperature_all", "value": float(value)})
                except (TypeError, ValueError):
                    print(f"[ORCHESTRATOR] adjusted_value 변환 실패: {value!r} — 적용 생략")
            decisions.append(
                f"크로스존 {action.get('type')} 승인 (값={value}) / {reason}"
            )
            if judgment.get("announcement"):
                print(f"[ORCHESTRATOR] 안내: {judgment['announcement']}")
        else:
            decisions.append(f"크로스존 요청 거부: {reason}")

    # 2. 충돌/불확실 고위험 — 알림 발송 의도 발행
    if state.get("conflict_detected"):
        result_dict = state.get("analysis_result") or {}
        if result_dict:
            actions.append({"kind": "alert", "result": result_dict})
            decisions.append(f"충돌 감지: {result_dict['detection_type']} 관리자 알림 발송")
            print(f"[ORCHESTRATOR] 충돌 감지 → 관리자 알림: {zone_id}")

    # 3. 보고서 이상 패턴 — 관리자 푸시 의도 발행
    if state.get("anomaly_detected") and state.get("report_text"):
        preview = (state["report_text"] or "")[:200]
        actions.append({"kind": "push", "title": "일일 운영 이상 감지", "body": preview})
        decisions.append("보고서 이상 패턴: 관리자 알림 발송")
        print("[ORCHESTRATOR] 보고서 이상 → 관리자 푸시 발송")

    return {
        "orchestrator_decision": "; ".join(decisions) if decisions else "처리 완료",
        "escalate_to_human": False,
        "pending_actions": actions,
    }
