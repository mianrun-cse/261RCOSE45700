"""
고객봇 에이전트 노드.
- 안전 우선순위를 시스템 프롬프트에 내재화 (안전 > 고객 편의 > 환경 최적화).
- "전체/모든 타석" 키워드 감지 시 cross_bay_request 설정 → 오케스트레이터 위임.
- 단일 베이 요청은 즉시 실행.
"""
import asyncio
import json
import pathlib
from openai import OpenAI

from llm_module.state import FacilityState
from llm_module.temperature_controller import handle as temp_handle
from llm_module.coaching_engine import _tts

client = OpenAI()
AUDIO_DIR = pathlib.Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
당신은 무인 스크린골프장의 AI 안내 도우미입니다.
우선순위 원칙: 안전 > 고객 편의 > 환경 최적화.
안전과 관련된 요청이 있으면 반드시 안전을 최우선으로 처리하세요.
고객의 요청에 친절하고 간결하게 한국어로 답변하세요.
환경 제어(온도, 조명, 팬)가 필요한 경우 반드시 JSON action을 포함하세요.
답변은 두 부분으로 구성하세요:
1. "message": 고객에게 보여줄 자연스러운 안내 문장
2. "action": 실행할 환경 제어 명령 (없으면 null)

action 형식:
{
  "type": "temperature" | "fan" | "light" | "extend_time" | "none",
  "value": <값>
}

반드시 JSON으로만 응답하세요.
"""

_CROSS_BAY_KEYWORDS = ["전체", "모든 타석", "전관", "전부", "모든 베이"]


async def customer_node(state: FacilityState) -> dict:
    bay_id = state["bay_id"]
    user_message = state.get("user_message") or ""
    context = state.get("customer_context") or {}

    context_str = json.dumps(context, ensure_ascii=False)
    user_content = f"[현재 상황]\n{context_str}\n\n[고객 요청]\n{user_message}"

    raw = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=200,
    )

    data = json.loads(raw.choices[0].message.content)
    message = data.get("message", "")
    action = data.get("action")

    # 시설 전체 영향 요청 감지 → 오케스트레이터 위임
    cross_bay = None
    if any(kw in user_message for kw in _CROSS_BAY_KEYWORDS):
        cross_bay = {"action": action}

    # 단일 베이 요청은 즉시 실행
    if action and action.get("type") != "none" and not cross_bay:
        await _execute_action(action, bay_id, context)

    audio_path = None
    if message and state.get("tts_enabled", True):
        filename = f"bot_{bay_id}_{abs(hash(message))}.mp3"
        audio_path = str(await _tts(message, filename))

    print(f"[{bay_id}][BOT] {message}")
    return {
        "bot_response": {"message": message, "audio_path": audio_path, "action": action},
        "cross_bay_request": cross_bay,
    }


async def _execute_action(action: dict, bay_id: str, context: dict) -> None:
    action_type = action.get("type")
    value = action.get("value")

    if action_type == "temperature":
        current = context.get("current_temp", 24.0)
        delta = float(value) - current
        await temp_handle(bay_id, current, 0, reason="고객 요청", target_temp_delta=delta)
    # fan, light → 환경 API 확장 시 추가
