from typing import TypedDict, Optional, List


class FacilityState(TypedDict, total=False):
    # 공통 (필수)
    bay_id: str
    trigger_type: str        # "safety" | "customer" | "coaching" | "report"
    all_bay_ids: List[str]

    # 안전 에이전트 입력
    analysis_result: Optional[dict]  # AnalysisResult 직렬화 dict
    signals: Optional[dict]          # TriggerSignals dict (temperature, humidity 등)

    # 고객봇 에이전트 입력/출력
    user_message: Optional[str]
    customer_context: Optional[dict]
    tts_enabled: bool                # False면 TTS 생성 생략
    bot_response: Optional[dict]     # {message, audio_path, action}

    # 코칭 에이전트 입력/출력
    pose_keypoints: Optional[dict]
    coaching_audio_path: Optional[str]

    # 보고서 에이전트 출력
    report_text: Optional[str]
    anomaly_detected: bool

    # 오케스트레이터
    cross_bay_request: Optional[dict]   # {action: {...}} 시설 전체 영향 요청
    conflict_detected: bool             # 에이전트 간 판단 충돌
    escalate_to_human: bool             # 관리자 에스컬레이션 필요
    orchestrator_decision: Optional[str]


def make_safety_state(
    bay_id: str,
    all_bay_ids: List[str],
    analysis_result: dict,
    signals: dict,
) -> FacilityState:
    return FacilityState(
        bay_id=bay_id,
        trigger_type="safety",
        all_bay_ids=all_bay_ids,
        analysis_result=analysis_result,
        signals=signals,
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_customer_state(
    bay_id: str,
    all_bay_ids: List[str],
    user_message: str,
    customer_context: dict,
    tts_enabled: bool = True,
) -> FacilityState:
    return FacilityState(
        bay_id=bay_id,
        trigger_type="customer",
        all_bay_ids=all_bay_ids,
        user_message=user_message,
        customer_context=customer_context,
        tts_enabled=tts_enabled,
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_coaching_state(
    bay_id: str,
    all_bay_ids: List[str],
    pose_keypoints: dict,
) -> FacilityState:
    return FacilityState(
        bay_id=bay_id,
        trigger_type="coaching",
        all_bay_ids=all_bay_ids,
        pose_keypoints=pose_keypoints,
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_report_state(
    bay_id: str,
    all_bay_ids: List[str],
) -> FacilityState:
    return FacilityState(
        bay_id=bay_id,
        trigger_type="report",
        all_bay_ids=all_bay_ids,
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )
