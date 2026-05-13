"""
고객 응대 챗봇 — LangGraph facility_graph 기반.
- 고객 요청을 customer 에이전트 노드로 라우팅
- 크로스베이 요청은 오케스트레이터가 자동 처리
- closing_notice / extension_offer는 단순 TTS라 그래프 미사용
"""
import asyncio
import pathlib
from llm_module.coaching_engine import _tts

AUDIO_DIR = pathlib.Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)


async def respond(
    user_message: str,
    bay_id: str,
    context: dict,
    tts: bool = True,
    all_bay_ids: list[str] | None = None,
) -> dict:
    """
    user_message: 고객이 입력한 텍스트 or 음성 인식 결과
    context: {
        "customer_name": str,
        "visit_count": int,
        "current_temp": float,
        "current_hole": int,
        "remaining_min": int,
        "reserved_min": int,
    }
    반환: {"message": str, "audio_path": str | None, "action": dict | None}
    """
    from llm_module.graph import facility_graph
    from llm_module.state import make_customer_state

    state = make_customer_state(
        bay_id=bay_id,
        all_bay_ids=all_bay_ids or [bay_id],
        user_message=user_message,
        customer_context=context,
    )

    result_state = await facility_graph.ainvoke(state)
    bot_response = result_state.get("bot_response") or {}

    # tts=False 요청 시 오디오 경로 제거
    if not tts:
        bot_response = {**bot_response, "audio_path": None}

    return bot_response


async def closing_notice(bay_id: str, remaining_min: int, context: dict) -> dict:
    """이용 종료 N분 전 자동 안내 (그래프 미사용 — 단순 TTS)"""
    msg = f"안내 말씀드립니다. {remaining_min}분 후 이용 시간이 종료됩니다. 연장을 원하시면 '연장'이라고 말씀해주세요."
    filename = f"closing_{bay_id}_{remaining_min}.mp3"
    audio_path = str(await _tts(msg, filename))
    return {"message": msg, "audio_path": audio_path, "action": None}


async def extension_offer(bay_id: str, current_hole: int, context: dict) -> dict:
    """게임 진행 상태 기반 연장 제안"""
    return await respond(
        f"현재 {current_hole}홀 진행 중이고 시간이 거의 끝나갑니다. 연장을 안내해주세요.",
        bay_id,
        context,
    )
