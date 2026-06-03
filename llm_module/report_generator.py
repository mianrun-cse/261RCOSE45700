import asyncio
import os
from datetime import datetime
from openai import OpenAI
from db.models import get_today_events, get_today_env_logs

client = OpenAI()


async def generate_daily_report(zone_ids: list[str]) -> str:
    """
    오늘 하루 운영 데이터를 LLM이 요약해 관리자용 리포트 생성.
    반환값: 리포트 텍스트 (문자/이메일 발송에 사용)
    """
    events   = await get_today_events()
    env_logs = []
    for zone_id in zone_ids:
        env_logs += await get_today_env_logs(zone_id)

    today = datetime.now().strftime("%Y년 %m월 %d일")

    event_summary = _summarize_events(events)
    env_summary   = _summarize_env(env_logs, zone_ids)

    prompt = f"""
다음은 무인 매장의 오늘({today}) 운영 데이터입니다.
관리자에게 보낼 간결한 일일 리포트를 한국어로 작성해주세요.
이상 있는 사항은 굵게 표시하고, 정상 운영된 사항은 간략히 기재하세요.

[이벤트 요약]
{event_summary}

[환경 제어 요약]
{env_summary}
"""

    response = await asyncio.to_thread(
        client.responses.create,
        model=os.getenv("REPORT_MODEL", "gpt-5-mini"),
        instructions="당신은 무인 매장 운영 관리 AI입니다.",
        input=prompt,
        service_tier="flex",
        store=False,
    )

    report = response.output_text.strip()
    print(f"[REPORT]\n{report}")
    return report


def _summarize_events(events: list[dict]) -> str:
    if not events:
        return "이벤트 없음"

    lines = []
    type_counts: dict[str, int] = {}
    alerted_events = []

    for e in events:
        type_counts[e["event_type"]] = type_counts.get(e["event_type"], 0) + 1
        if e["alerted"]:
            alerted_events.append(
                f"  - [{e['occurred_at'][11:16]}] {e['zone_id']} / {e['event_type']} "
                f"(신뢰도 {e['confidence']:.0%}): {e['evidence']}"
            )

    for et, count in type_counts.items():
        lines.append(f"{et}: {count}건")

    if alerted_events:
        lines.append("\n[관리자 알림 발송 건]")
        lines += alerted_events

    return "\n".join(lines)


def _summarize_env(env_logs: list[dict], zone_ids: list[str]) -> str:
    if not env_logs:
        return "환경 제어 없음"

    by_zone: dict[str, list] = {z: [] for z in zone_ids}
    for log in env_logs:
        if log["zone_id"] in by_zone:
            by_zone[log["zone_id"]].append(log)

    lines = []
    for zone_id, logs in by_zone.items():
        if not logs:
            lines.append(f"{zone_id}: 제어 없음")
            continue
        temps = [l["temperature"] for l in logs if l["temperature"]]
        avg_t = sum(temps) / len(temps) if temps else 0
        lines.append(f"{zone_id}: 자동 제어 {len(logs)}회 / 평균 설정 온도 {avg_t:.1f}도")

    return "\n".join(lines)
