"""
무인 매장 LLM 모듈 진입점.

카메라 인덱스 매핑:
  ZONE_CAMERAS = {"1번 구역": 0, "2번 구역": 1, ...}
  카메라가 없는 구역은 매핑에서 제외하면 bridge가 실행되지 않는다.
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from db.models import init_db
from llm_module.state_machine import ZoneStateMachine, TriggerSignals
from opencv.bridge import run as cv_run

# ── 설정 ──────────────────────────────────────────────────────────────────────

ZONE_IDS   = ["1번 구역", "2번 구역", "3번 구역"]
CLOSE_HOUR = int(os.getenv("CLOSE_HOUR", 22))

# 구역별 카메라 인덱스 (없는 구역은 제외)
ZONE_CAMERAS: dict[str, int] = {
    "1번 구역": 0,
    "2번 구역": 1,
    "3번 구역": 2,
}

# ── 구역별 큐 ─────────────────────────────────────────────────────────────────

frame_queues:  dict[str, asyncio.Queue] = {}
signal_queues: dict[str, asyncio.Queue] = {}


# ── 루프 ──────────────────────────────────────────────────────────────────────

async def zone_loop(zone_id: str) -> None:
    machine = ZoneStateMachine(zone_id, frame_queues[zone_id], all_zone_ids=ZONE_IDS)
    while True:
        try:
            signals: TriggerSignals = signal_queues[zone_id].get_nowait()
        except asyncio.QueueEmpty:
            signals = TriggerSignals()
        await machine.update(signals)
        await asyncio.sleep(0.05)


async def report_loop() -> None:
    import datetime
    from llm_module.graph import facility_graph
    from llm_module.state import make_report_state

    while True:
        now = datetime.datetime.now()
        if now.hour == CLOSE_HOUR and now.minute == 0:
            state = make_report_state(zone_id=ZONE_IDS[0], all_zone_ids=ZONE_IDS)
            result = await facility_graph.ainvoke(state)
            print("[DAILY REPORT]\n", result.get("report_text", ""))
            await asyncio.sleep(60)
        await asyncio.sleep(30)


async def main() -> None:
    await init_db()
    print(f"[MAIN] DB 초기화 완료. 구역: {ZONE_IDS}")

    for zone_id in ZONE_IDS:
        frame_queues[zone_id]  = asyncio.Queue(maxsize=1)
        signal_queues[zone_id] = asyncio.Queue(maxsize=1)

    tasks = [zone_loop(z) for z in ZONE_IDS] + [report_loop()]

    # OpenCV 브릿지 — 카메라가 연결된 구역만 실행
    for zone_id, cam_idx in ZONE_CAMERAS.items():
        tasks.append(
            cv_run(zone_id, frame_queues[zone_id], signal_queues[zone_id], cam_idx)
        )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
