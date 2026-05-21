"""
카메라 없이 각 모듈을 독립적으로 테스트하는 스크립트.
실행: python test_modules.py
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()


async def test_customer_bot():
    print("\n" + "="*50)
    print("[ 고객 응대 챗봇 테스트 ]")
    print("="*50)
    from llm_module.customer_bot import respond, closing_notice, extension_offer

    context = {
        "customer_name": "김철수",
        "visit_count": 3,
        "current_temp": 26.0,
        "remaining_min": 10,
        "reserved_min": 60,
    }

    tests = [
        "좀 더 춥게 해줘",
        "남은 시간 알려줘",
        "시간 연장하고 싶어",
        "너무 더워요",
    ]

    for msg in tests:
        print(f"\n고객: {msg}")
        result = await respond(msg, "1번 구역", context, tts=False)
        print(f"봇:   {result['message']}")
        if result["action"]:
            print(f"액션: {result['action']}")


async def test_report_generator():
    print("\n" + "="*50)
    print("[ 일일 리포트 테스트 ]")
    print("="*50)
    from llm_module.report_generator import generate_daily_report

    report = await generate_daily_report(["1번 구역", "2번 구역", "3번 구역"])
    print(report)


async def test_alert_manager():
    print("\n" + "="*50)
    print("[ 관리자 알림 테스트 (SMS/푸시 미설정 시 콘솔 출력) ]")
    print("="*50)
    from llm_module.vlm_analyzer import AnalysisResult, DetectionType, Severity
    from llm_module.alert_manager import handle

    # 도난 감지 시뮬레이션
    result = AnalysisResult(
        detection_type=DetectionType.THEFT,
        detected=True,
        confidence=0.88,
        severity=Severity.HIGH,
        evidence="Person moving merchandise toward exit",
        action_required="Notify manager immediately",
    )
    print("\n도난 감지 알림 테스트:")
    await handle(result, "2번 구역")

    # 땀 감지 시뮬레이션
    result2 = AnalysisResult(
        detection_type=DetectionType.SWEAT_WIPING,
        detected=True,
        confidence=0.91,
        severity=Severity.LOW,
        evidence="Hand touching forehead repeatedly",
        action_required="Lower temperature",
    )
    print("\n땀 감지 알림 테스트:")
    await handle(result2, "1번 구역")


async def test_multi_agent():
    """
    멀티에이전트 협력 시나리오 테스트.
    [AGENT: xxx] 로그로 어떤 에이전트가 실행됐는지 확인 가능.
    """
    print("\n" + "="*50)
    print("[ 멀티에이전트 시나리오 테스트 ]")
    print("="*50)
    from llm_module.graph import facility_graph
    from llm_module.state import make_customer_state, make_safety_state

    ZONE_IDS = ["1번 구역", "2번 구역", "3번 구역"]

    # ── 시나리오 1: 단일 구역 요청 → customer → actuator ──────────────────────
    print("\n[시나리오 1] 단일 구역 요청 (오케스트레이터 미개입 예상)")
    print("  기대 경로: customer → actuator → END")
    state = make_customer_state("1번 구역", ZONE_IDS, "온도 좀 낮춰줘", {"current_temp": 27.0}, tts_enabled=False)
    result = await facility_graph.ainvoke(state)
    print(f"  응답: {result.get('bot_response', {}).get('message', '')}")
    print(f"  오케스트레이터 결정: {result.get('orchestrator_decision', '(미실행)')}")

    # ── 시나리오 2: 전체 구역 요청 → customer → orchestrator → actuator ───────
    print("\n[시나리오 2] 전체 구역 요청 (오케스트레이터 개입 예상)")
    print("  기대 경로: customer → orchestrator → actuator → END")
    state2 = make_customer_state("1번 구역", ZONE_IDS, "전체 구역 온도 22도로 맞춰줘", {"current_temp": 27.0}, tts_enabled=False)
    result2 = await facility_graph.ainvoke(state2)
    print(f"  응답: {result2.get('bot_response', {}).get('message', '')}")
    print(f"  오케스트레이터 결정: {result2.get('orchestrator_decision', '(미실행)')}")

    # ── 시나리오 3: 고위험 불확실 안전 감지 → safety → orchestrator → actuator ─
    print("\n[시나리오 3] 고위험 불확실 감지 (오케스트레이터 개입 예상)")
    print("  기대 경로: safety → orchestrator → actuator → END")
    state3 = make_safety_state(
        zone_id="2번 구역",
        all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": "fall_emergency",
            "detected": True,
            "confidence": 0.72,   # 0.80 미만 → 오케스트레이터 위임
            "severity": "high",
            "evidence": "Person lying on floor, unclear if intentional",
            "action_required": "Verify with customer or manager",
        },
        signals={"temperature": 25.0, "humidity": 60.0},
    )
    result3 = await facility_graph.ainvoke(state3)
    print(f"  충돌 감지됨: {result3.get('conflict_detected', False)}")
    print(f"  오케스트레이터 결정: {result3.get('orchestrator_decision', '(미실행)')}")


# ── 메뉴 ──────────────────────────────────────────────────────────────────────

TESTS = {
    "1": ("고객 응대 챗봇",         test_customer_bot),
    "2": ("일일 리포트",             test_report_generator),
    "3": ("관리자 알림",             test_alert_manager),
    "4": ("멀티에이전트 시나리오",    test_multi_agent),
    "5": ("전체 실행",               None),
}

async def main():
    from db.models import init_db
    await init_db()

    print("\n테스트할 모듈을 선택하세요:")
    for k, (name, _) in TESTS.items():
        print(f"  [{k}] {name}")

    choice = input("\n선택: ").strip()

    if choice == "5":
        for k, (name, fn) in TESTS.items():
            if fn:
                await fn()
    elif choice in TESTS:
        _, fn = TESTS[choice]
        await fn()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    asyncio.run(main())
