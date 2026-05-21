import operator
from typing import TypedDict, Optional, List, Annotated


class FacilityState(TypedDict, total=False):
    # 공통 (필수)
    zone_id: str
    trigger_type: str        # "safety" | "customer" | "report"
    all_zone_ids: List[str]

    # 결정 노드가 발행하는 실행 의도. actuator 노드가 일괄 실행한다.
    # 여러 노드의 발행이 누적되도록 add 리듀서를 사용한다.
    pending_actions: Annotated[List[dict], operator.add]

    # 안전 에이전트 입력
    analysis_result: Optional[dict]  # AnalysisResult 직렬화 dict
    signals: Optional[dict]          # TriggerSignals dict (temperature, humidity 등)

    # 고객봇 에이전트 입력/출력
    user_message: Optional[str]
    customer_context: Optional[dict]
    tts_enabled: bool                # False면 TTS 생성 생략
    bot_response: Optional[dict]     # {message, audio_path, action}

    # 보고서 에이전트 출력
    report_text: Optional[str]
    anomaly_detected: bool

    # 오케스트레이터
    cross_zone_request: Optional[dict]  # {action: {...}} 시설 전체 영향 요청
    conflict_detected: bool             # 에이전트 간 판단 충돌
    escalate_to_human: bool             # 관리자 에스컬레이션 필요
    orchestrator_decision: Optional[str]


def make_safety_state(
    zone_id: str,
    all_zone_ids: List[str],
    analysis_result: dict,
    signals: dict,
) -> FacilityState:
    return FacilityState(
        zone_id=zone_id,
        trigger_type="safety",
        all_zone_ids=all_zone_ids,
        analysis_result=analysis_result,
        signals=signals,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_customer_state(
    zone_id: str,
    all_zone_ids: List[str],
    user_message: str,
    customer_context: dict,
    tts_enabled: bool = True,
) -> FacilityState:
    return FacilityState(
        zone_id=zone_id,
        trigger_type="customer",
        all_zone_ids=all_zone_ids,
        user_message=user_message,
        customer_context=customer_context,
        tts_enabled=tts_enabled,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_report_state(
    zone_id: str,
    all_zone_ids: List[str],
) -> FacilityState:
    return FacilityState(
        zone_id=zone_id,
        trigger_type="report",
        all_zone_ids=all_zone_ids,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )
