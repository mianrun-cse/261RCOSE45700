"""
인사이트 에이전트 노드 (결정 전용).

상권 인구통계 프로필(DistrictProfile)을 LLM이 해석해, 무인 매장 운영자를 위한
자연어 인사이트(피크 시간대·수요 특성·적합 업종)를 생성한다.
부수효과 없음 — 결과는 state["insight_result"]에만 기록하고 추천 노드로 넘긴다.

기존 report_agent.py(데이터→LLM→텍스트) 패턴을 인사이트 추천 엔진으로 전환한 노드.
"""
import asyncio
import json
import os

from openai import OpenAI

from llm_module.state import FacilityState

client = OpenAI()

_SYSTEM_PROMPT = """
당신은 무인 매장 입지·운영 전략을 분석하는 상권 인텔리전스 AI입니다.
주어진 상권의 인구통계 프로필(연령 분포, 성비, 직업군, 가구형태, 주거형태, 학력, 취미)을
해석해 무인 매장 운영자/창업자에게 도움이 되는 인사이트를 도출하세요.

분석 지침:
- 직업군과 연령 분포로 시간대별 유동·수요 패턴을 추론하세요
  (예: 직장인 밀집 → 출근 07-09시·점심 12-13시 피크 / 1인 가구 多 → 야간·심야 생활수요).
- hourly_footfall이 주어지면 그 값을 우선 근거로 삼고, 없으면 인구통계로 추론하세요.
- 추론에는 반드시 프로필의 구체적 수치를 근거로 인용하세요.
- 무인 매장 업태(무인 카페, 무인 편의점, 무인 세탁소, 스낵·음료 자판기, 무인 아이스크림 등) 관점으로 좁히세요.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "summary": "이 상권을 한 문장으로 요약",
  "insights": [
    {"category": "유동/시간대" | "수요" | "업종적합도" | "리스크",
     "insight": "근거 수치를 포함한 자연어 인사이트 한 문장",
     "confidence": "high" | "medium" | "low"}
  ],
  "peak_hours": ["07:00-09:00", "22:00-01:00"],
  "recommended_store_types": ["고단백 스낵 자판기", "무인 세탁소"]
}
"""


async def insight_node(state: FacilityState) -> dict:
    profile = state.get("district_profile") or {}
    district = profile.get("district", state.get("zone_id", "unknown"))
    print(f"[AGENT: insight] district={district} sample={profile.get('sample_size')}")

    user_content = (
        "다음은 한 상권의 인구통계 프로필(JSON)입니다. "
        "이를 해석해 무인 매장 운영 인사이트를 JSON으로 도출하세요.\n\n"
        f"{json.dumps(profile, ensure_ascii=False)}"
    )

    raw = await asyncio.to_thread(
        client.responses.create,
        model=os.getenv("INSIGHT_MODEL", "gpt-5-mini"),
        instructions=_SYSTEM_PROMPT,
        input=user_content,
        text={"format": {"type": "json_object"}},
        service_tier="default",
        store=False,
    )

    result = json.loads(raw.output_text)
    summary = result.get("summary", "")
    print(f"[{district}][INSIGHT] {summary}")

    return {"insight_result": result}
