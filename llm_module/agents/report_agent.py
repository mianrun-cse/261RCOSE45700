"""
보고서 에이전트 노드.
- 일일 리포트 생성 후 이상 패턴(위험/도난/파손 키워드) 감지 시 오케스트레이터에 보고.
- 분석과 인사이트 생성만 담당; 알림 발송은 오케스트레이터 책임.
"""
from llm_module.state import FacilityState
from llm_module.report_generator import generate_daily_report

_ANOMALY_KEYWORDS = ["이상", "위험", "응급", "도난", "파손", "주의", "심각"]


async def report_node(state: FacilityState) -> dict:
    print(f"[AGENT: report] bays={state.get('all_bay_ids')}")
    all_bay_ids = state.get("all_bay_ids") or []
    report = await generate_daily_report(all_bay_ids)

    anomaly = any(kw in report for kw in _ANOMALY_KEYWORDS)

    return {
        "report_text": report,
        "anomaly_detected": anomaly,
        "escalate_to_human": anomaly,
    }
