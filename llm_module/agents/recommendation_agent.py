"""
추천 에이전트 노드 (결정 전용).

상권 프로필 + 인사이트 노드의 분석 결과를 받아, 무인 매장 운영자에게
구체적인 상품 구성(product mix)과 재고 조정 권장안을 생성한다.
부수효과 없음 — 결과는 state["recommendations"]에만 기록한다.

insight_node → recommendation_node 순서로 실행되며, 인사이트 노드의 출력을
입력으로 받아 "무엇을 얼마나 입고할지" 수준으로 구체화한다.
"""
import asyncio
import json
import os

from openai import OpenAI

from llm_module.state import FacilityState

client = OpenAI()

_SYSTEM_PROMPT = """
당신은 무인 매장 상품 구성(MD) 추천 AI입니다.
상권 인구통계 프로필과 사전 분석된 인사이트를 근거로, 운영자가 바로 실행할 수 있는
구체적인 상품 구성과 재고 조정안을 제시하세요.

지침:
- 상품은 무인 매장에서 실제 판매 가능한 품목으로 한정하세요
  (음료, 스낵, 간편식, 생활용품, 세탁/충전 등 무인 서비스).
- 각 상품에 수요 등급과 피크 시간대, 재고 조정 권장치를 제시하세요.
- 인사이트의 피크 시간대·수요 특성과 일관되게 작성하세요.
- 권장 근거(rationale)에는 인구통계 수치나 인사이트를 인용하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "location": "상권 이름",
  "products": [
    {"name": "콜드브루 커피",
     "demand": "high" | "moderate" | "steady" | "low",
     "peak_window": "07:00-09:00",
     "stock_adjustment": "+30%",
     "rationale": "근거 한 문장"}
  ],
  "store_type_fit": "이 상권에 가장 적합한 무인 매장 업태 한 문장"
}
"""


async def recommendation_node(state: FacilityState) -> dict:
    profile = state.get("district_profile") or {}
    insight = state.get("insight_result") or {}
    district = profile.get("district", state.get("zone_id", "unknown"))
    print(f"[AGENT: recommendation] district={district}")

    user_content = (
        "아래 상권 프로필과 인사이트를 근거로 무인 매장 상품 구성 추천을 JSON으로 작성하세요.\n\n"
        f"[상권 프로필]\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"[사전 인사이트]\n{json.dumps(insight, ensure_ascii=False)}"
    )

    raw = await asyncio.to_thread(
        client.responses.create,
        model=os.getenv("RECOMMENDATION_MODEL", "gpt-5-mini"),
        instructions=_SYSTEM_PROMPT,
        input=user_content,
        text={"format": {"type": "json_object"}},
        service_tier="default",
        store=False,
    )

    recommendations = json.loads(raw.output_text)
    n_products = len(recommendations.get("products", []))
    print(f"[{district}][RECOMMEND] {n_products}개 품목 / {recommendations.get('store_type_fit', '')}")

    return {"recommendations": recommendations}
