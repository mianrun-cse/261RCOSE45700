"""
안전 감지 에이전트 노드.
vlm_analyzer의 분석 결과를 받아 알림/온도 제어 실행 또는 오케스트레이터 에스컬레이션 결정.
- gpt-4o-mini 기본, confidence < 0.7 시 gpt-4o로 자율 에스컬레이션.
- 고위험 + confidence < 0.80 → conflict_detected=True → 오케스트레이터 판단 위임.
"""
import asyncio
import base64
import json
import os

from llm_module.state import FacilityState
from llm_module.alert_manager import handle as alert_handle
from llm_module.temperature_controller import handle as temp_handle
from llm_module.vlm_analyzer import (
    AnalysisResult, DetectionType, Severity,
    _CONFIGS, _build_input, encode_frame,
)

ESCALATION_MODEL = os.getenv("ESCALATION_MODEL", "gpt-4o")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
ESCALATION_THRESHOLD = 0.70  # mini 결과가 이 미만이면 4o로 재판단


async def _reanalyze_with_better_model(
    frames_b64: list[str],
    detection_type: DetectionType,
) -> AnalysisResult:
    """confidence 낮은 케이스를 더 좋은 모델로 재판단."""
    from openai import OpenAI
    client = OpenAI()
    cfg = _CONFIGS[detection_type]

    raw = await asyncio.to_thread(
        client.responses.create,
        model=ESCALATION_MODEL,
        input=_build_input(cfg["prompt"], frames_b64),
        text={
            "format": {
                "type": "json_schema",
                "name": cfg["schema_name"],
                "strict": True,
                "schema": cfg["schema"],
            }
        },
        max_output_tokens=200,
        store=False,
    )

    data = json.loads(raw.output_text)
    detected = data["detected"] and data["confidence"] >= CONFIDENCE_THRESHOLD
    return AnalysisResult(
        detection_type=detection_type,
        detected=detected,
        confidence=data["confidence"],
        severity=cfg["severity"],
        evidence=data["evidence"],
        action_required=data["action_required"],
    )


async def safety_node(state: FacilityState) -> dict:
    result_dict = state.get("analysis_result") or {}
    if not result_dict:
        return {}

    signals = state.get("signals") or {}
    bay_id = state["bay_id"]

    detection_type = DetectionType(result_dict["detection_type"])
    result = AnalysisResult(
        detection_type=detection_type,
        detected=result_dict["detected"],
        confidence=result_dict["confidence"],
        severity=Severity(result_dict["severity"]),
        evidence=result_dict["evidence"],
        action_required=result_dict["action_required"],
    )

    # 모델 자율 에스컬레이션: confidence 낮으면 더 큰 모델로 재판단
    if result.detected and result.confidence < ESCALATION_THRESHOLD:
        frames_b64 = result_dict.get("frames_b64") or []
        if frames_b64:
            print(f"[{bay_id}][SAFETY] confidence={result.confidence:.2f} → {ESCALATION_MODEL}로 재판단")
            result = await _reanalyze_with_better_model(frames_b64, detection_type)

    if not result.detected:
        return {}

    # 고위험 + 여전히 불확실 → 오케스트레이터 판단 위임
    if result.severity == Severity.HIGH and result.confidence < 0.80:
        print(f"[{bay_id}][SAFETY] 고위험 불확실({result.confidence:.2f}) → 오케스트레이터 위임")
        return {"conflict_detected": True}

    # 알림 발송 (tool)
    await alert_handle(result, bay_id)

    # 땀 감지 → 온도 제어 (tool)
    if result.detection_type == DetectionType.SWEAT_WIPING:
        await temp_handle(
            bay_id=bay_id,
            temperature=signals.get("temperature", 25.0),
            humidity=signals.get("humidity", 60.0),
            reason=result.evidence,
        )

    return {}
