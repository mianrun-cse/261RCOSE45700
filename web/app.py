"""
멀티에이전트 흐름 시각화 웹 서버.

브라우저는 /stream(SSE) 한 채널만 구독하고, 그래프 실행 이벤트는
어떤 경로로 들어오든 같은 broadcast 채널로 fan-out된다.

이벤트 소스 두 가지:
  1) 웹 페이지의 버튼 → POST /trigger/{key} → 서버가 그래프 직접 실행
  2) 외부 클라이언트(test_agent.py --web-url) → POST /events

브라우저:
  GET /stream (text/event-stream) - 모든 이벤트 fan-out 수신

actuator의 부수효과(SMS/온도 제어/TTS)는 데모 안전을 위해 모킹된다.

실행:
  pip install -r requirements.txt
  py -m uvicorn web.app:app --reload --port 8000
  → http://localhost:8000
"""
import asyncio
import contextlib
import io
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from db.models import init_db
from llm_module.state import (
    make_customer_state, make_safety_state, make_report_state, make_insight_state,
)
from data_simulation import SAMPLE_PROFILES
from test_agent import install_actuator_mocks

ZONE_IDS = ["1번 구역", "2번 구역", "3번 구역"]

app = FastAPI(title="무인 매장 멀티에이전트 흐름")

STATIC_DIR = Path(__file__).parent / "static"

_started = False


@app.on_event("startup")
async def _startup():
    global _started
    if _started:
        return
    await init_db()
    install_actuator_mocks()
    _started = True
    print("[WEB] 시작 - actuator 모킹 완료, DB 초기화 완료")


# ── Broadcast 채널 ────────────────────────────────────────────────────────────

_subscribers: list[asyncio.Queue] = []
_subs_lock = asyncio.Lock()


async def broadcast(event: str, data: dict) -> None:
    """모든 SSE subscriber에게 이벤트 fan-out."""
    payload = (event, data)
    async with _subs_lock:
        dead: list[asyncio.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


@app.get("/stream")
async def stream():
    """브라우저용 SSE 채널. 페이지 로드 시 자동 연결되어 모든 이벤트 수신."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    async with _subs_lock:
        _subscribers.append(queue)

    async def gen():
        try:
            yield _sse("hello", {"subscribers": len(_subscribers)})
            while True:
                event, data = await queue.get()
                yield _sse(event, data)
        except asyncio.CancelledError:
            pass
        finally:
            async with _subs_lock:
                if queue in _subscribers:
                    _subscribers.remove(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 외부 이벤트 수신 (test_agent → web) ──────────────────────────────────────

class ExternalEvent(BaseModel):
    event: str
    data: dict


@app.post("/events")
async def post_event(evt: ExternalEvent):
    if evt.event not in ("start", "log", "done", "error"):
        raise HTTPException(400, f"unknown event: {evt.event}")
    await broadcast(evt.event, evt.data)
    return {"ok": True}


# ── 페이지에서 직접 트리거 ───────────────────────────────────────────────────

def _customer_single():
    return make_customer_state(
        zone_id="1번 구역", all_zone_ids=ZONE_IDS,
        user_message="온도 좀 낮춰줘",
        customer_context={"customer_name": "데모", "visit_count": 1,
                          "current_temp": 27.0, "remaining_min": 30, "reserved_min": 60},
        tts_enabled=False,
    )


def _customer_cross():
    return make_customer_state(
        zone_id="1번 구역", all_zone_ids=ZONE_IDS,
        user_message="전체 구역 온도 22도로 맞춰줘",
        customer_context={"customer_name": "데모", "visit_count": 1,
                          "current_temp": 26.0, "remaining_min": 40, "reserved_min": 60},
        tts_enabled=False,
    )


def _safety_conflict():
    return make_safety_state(
        zone_id="2번 구역", all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": "fall_emergency", "detected": True,
            "confidence": 0.72, "severity": "high",
            "evidence": "Person lying still, unclear if intentional",
            "action_required": "Verify",
        },
        signals={"temperature": 25.0, "humidity": 60.0},
    )


def _safety_normal():
    return make_safety_state(
        zone_id="2번 구역", all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": "theft", "detected": True,
            "confidence": 0.92, "severity": "high",
            "evidence": "Hand moving merchandise toward bag",
            "action_required": "Notify manager",
        },
        signals={"temperature": 24.0, "humidity": 55.0},
    )


def _report():
    return make_report_state(zone_id="1번 구역", all_zone_ids=ZONE_IDS)


def _insight():
    return make_insight_state(district_profile=SAMPLE_PROFILES["서울-강남구"])


SCENARIOS = {
    "customer_single":  ("고객 - 단일 구역 온도 요청",              _customer_single),
    "customer_cross":   ("고객 - 전체 구역 요청 (크로스존)",         _customer_cross),
    "safety_conflict":  ("안전 - 고위험 불확실 (충돌 라우팅)",       _safety_conflict),
    "safety_normal":    ("안전 - 명확한 도난 감지",                  _safety_normal),
    "report":           ("일일 보고서 생성",                          _report),
    "insight":          ("인사이트 - 상권 분석 (서울-강남구)",        _insight),
}

_NODE_LOG = re.compile(r"\[AGENT:\s*(\w+)\]")


@app.get("/scenarios")
async def list_scenarios():
    return [{"key": k, "name": name} for k, (name, _) in SCENARIOS.items()]


# ── 인사이트 엔진 API (운영자 대시보드 백엔드) ───────────────────────────────

class InsightRequest(BaseModel):
    """상권 인구통계 프로필(DistrictProfile). 미지정 필드는 LLM이 추론한다.

    district만으로 호출하면 SAMPLE_PROFILES에서 사전 집계 프로필을 사용한다
    (예: {"district": "서울-강남구"}). 전체 프로필을 직접 넘기면 그 값을 사용한다.
    """
    district: str | None = None
    profile: dict | None = None


@app.post("/insight")
async def insight(req: InsightRequest):
    """상권 프로필 → AI 인사이트 + 상품 추천 (동기 응답).

    그래프(insight → recommendation)를 한 번 실행하고 최종 결과만 반환한다.
    (실시간 흐름 시각화가 필요하면 페이지의 '인사이트' 버튼 = POST /trigger/insight 사용)
    """
    profile = req.profile
    if profile is None:
        if req.district and req.district in SAMPLE_PROFILES:
            profile = SAMPLE_PROFILES[req.district]
        else:
            raise HTTPException(
                400,
                "profile 또는 SAMPLE_PROFILES에 존재하는 district가 필요합니다. "
                f"사용 가능 district: {list(SAMPLE_PROFILES)}",
            )

    from llm_module.graph import facility_graph

    state = make_insight_state(district_profile=profile)
    result = await facility_graph.ainvoke(state)
    return {
        "district": profile.get("district"),
        "profile": profile,
        "insight": result.get("insight_result"),
        "recommendations": result.get("recommendations"),
    }

@app.get("/districts")  
async def get_districts():  
    return {"districts": list(SAMPLE_PROFILES.keys())} 

@app.post("/trigger/{key}")
async def trigger(key: str):
    """페이지의 시나리오 버튼이 호출. 백그라운드에서 그래프 실행하면서 이벤트 broadcast."""
    if key not in SCENARIOS:
        raise HTTPException(404, f"unknown scenario: {key}")

    state = SCENARIOS[key][1]()
    name = SCENARIOS[key][0]

    asyncio.create_task(_run_and_broadcast(name, state, source="web"))
    return {"ok": True}


def _input_from_state(state: dict) -> dict:
    """초기 state에서 입력 요약 추출 (웹 페이지 표시용)."""
    trigger = state.get("trigger_type")
    if trigger == "customer":
        return {
            "trigger_type": "customer",
            "zone_id": state.get("zone_id"),
            "user_message": state.get("user_message"),
            "context": state.get("customer_context") or {},
        }
    if trigger == "safety":
        return {
            "trigger_type": "safety",
            "zone_id": state.get("zone_id"),
            "analysis_result": state.get("analysis_result") or {},
            "signals": state.get("signals") or {},
        }
    if trigger == "report":
        return {
            "trigger_type": "report",
            "zone_id": state.get("zone_id"),
            "all_zone_ids": state.get("all_zone_ids") or [],
        }
    if trigger == "insight":
        return {
            "trigger_type": "insight",
            "district_profile": state.get("district_profile") or {},
        }
    return {"trigger_type": trigger}


async def _run_and_broadcast(name: str, state: dict, source: str) -> None:
    """그래프 실행 + stdout 캡처 → broadcast."""
    from llm_module.graph import facility_graph

    input_summary = _input_from_state(state)

    await broadcast("start", {
        "name": name,
        "source": source,
        "input": input_summary,
    })
    loop = asyncio.get_running_loop()
    line_q: asyncio.Queue = asyncio.Queue()

    class _Writer(io.TextIOBase):
        def __init__(self):
            self._buf = ""

        def write(self, s: str) -> int:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    loop.call_soon_threadsafe(line_q.put_nowait, line)
            return len(s)

    async def emit_lines():
        while True:
            line = await line_q.get()
            if line is None:
                return
            m = _NODE_LOG.search(line)
            await broadcast("log", {
                "node": m.group(1) if m else None,
                "line": line,
                "source": source,
                **input_summary,
            })

    emit_task = asyncio.create_task(emit_lines())
    try:
        with contextlib.redirect_stdout(_Writer()):
            result = await facility_graph.ainvoke(state)
        await asyncio.sleep(0.05)  # 마지막 라인이 큐로 flush 되도록
        await broadcast("done", {
            "name": name, "source": source,
            "input": input_summary,
            "state": _serialize_state(result),
        })
    except Exception as e:
        await broadcast("error", {
            "name": name, "source": source,
            "message": f"{type(e).__name__}: {e}",
        })
    finally:
        line_q.put_nowait(None)
        try:
            await asyncio.wait_for(emit_task, timeout=1.0)
        except asyncio.TimeoutError:
            emit_task.cancel()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _serialize_state(s: dict) -> dict:
    keep = (
        "trigger_type", "zone_id",
        "user_message", "analysis_result", "signals",
        "bot_response", "orchestrator_decision",
        "cross_zone_request", "conflict_detected", "anomaly_detected",
        "report_text", "pending_actions",
        "district_profile", "insight_result", "recommendations",
    )
    out: dict = {}
    for k in keep:
        if k in s and s[k] not in (None, [], {}, False):
            out[k] = s[k]
    return out


# ── 정적 페이지 ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/dashboard", response_class=HTMLResponse)  
async def dashboard():  
    return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")  

@app.get("/store", response_class=HTMLResponse)
async def store_page():
    return (STATIC_DIR / "store.html").read_text(encoding="utf-8")

# ── 카페 흐름 생성 API ────────────────────────────────────────────────────────

class CafeFlowRequest(BaseModel):
    district: str | None = None
    profile:  dict | None = None


@app.post("/cafe-flow")
async def cafe_flow(req: CafeFlowRequest):
    """
    상권 프로필 → LLM이 무인카페 고객 흐름(flow stages) + 메뉴 + 좌석 선호도 생성.
    store.html 시뮬레이션이 이 JSON을 그대로 사용한다.
    """
    profile = req.profile
    if profile is None:
        if req.district and req.district in SAMPLE_PROFILES:
            profile = SAMPLE_PROFILES[req.district]
        else:
            raise HTTPException(
                400,
                f"profile 또는 SAMPLE_PROFILES에 존재하는 district 필요. "
                f"사용 가능: {list(SAMPLE_PROFILES)}",
            )

    from llm_module.cafe_flow import generate_cafe_flow
    result = await generate_cafe_flow(profile)
    return result

# ── 고객 쾌적도 요청 (시뮬레이션 → 실제 에이전트 처리) ──────────────────────

class ComfortRequest(BaseModel):
    persona_name: str
    persona_age:  int
    zone_id:      str
    event_id:     str           # "too_hot" | "too_cold" | "stuffy"
    message:      str
    current_temp: float = 24.0
    # action is no longer sent from frontend — AI decides via full graph


@app.post("/persona-comfort")
async def persona_comfort(req: ComfortRequest):
    """
    store.html 시뮬레이션의 페르소나가 불쾌감을 느낄 때 호출.

    흐름: facility_graph 전체 실행 (customer → reconcile → actuator)
    AI가 메시지를 보고 온도 조정 여부와 delta를 직접 결정한다.
    """
    from llm_module.state import make_customer_state
    from llm_module.graph import facility_graph

    state = make_customer_state(
        zone_id=req.zone_id,
        all_zone_ids=ZONE_IDS,
        user_message=req.message,
        customer_context={
            "customer_name": req.persona_name,
            "visit_count":   1,
            "current_temp":  req.current_temp,
            "remaining_min": 30,
            "reserved_min":  60,
        },
        tts_enabled=False,
    )

    result = await facility_graph.ainvoke(state)

    bot_response = result.get("bot_response") or {}
    bot_message  = bot_response.get("message", "")

    # Extract AI-decided temperature change from pending_actions
    # Action format: {"kind":"temperature", "target_temp_delta": float, ...}
    ctrl_result = {"executed": False, "new_temp": req.current_temp}

    for act in result.get("pending_actions", []):
        kind = act.get("kind", "")
        if kind in ("temperature", "temperature_all"):
            delta    = float(act.get("target_temp_delta", 0))
            new_temp = round(max(18.0, min(30.0, req.current_temp + delta)), 1)
            ctrl_result = {
                "executed": True,
                "new_temp": new_temp,
                "delta":    delta,
                "type":     kind,
            }
            break

    # broadcast → store.html real-time update
    await broadcast("comfort_event", {
        "persona_name": req.persona_name,
        "persona_age":  req.persona_age,
        "zone_id":      req.zone_id,
        "event_id":     req.event_id,
        "message":      req.message,
        "bot_message":  bot_message,
        "ctrl_result":  ctrl_result,
    })

    return {
        "ok":          True,
        "bot_message": bot_message,
        "ctrl_result": ctrl_result,
    }


# ── 안전 이벤트 (시뮬레이션 페르소나 불안 신고) ──────────────────────────────

class SafetyReportRequest(BaseModel):
    persona_name:    str
    persona_age:     int
    zone_id:         str
    event_id:        str        # "suspicious_person" | "harassment"
    detection_type:  str
    confidence:      float
    severity:        str
    evidence:        str
    current_temp:    float = 24.0


@app.post("/persona-safety")
async def persona_safety(req: SafetyReportRequest):
    """
    store.html 시뮬레이션 페르소나가 불안을 느낄 때 호출.

    흐름:
      1) safety_node (기존 안전 에이전트) 실행
      2) broadcast("safety_alert") → store.html 경보 오버레이 표시
    """
    from llm_module.state import make_safety_state
    from llm_module.graph import facility_graph

    state = make_safety_state(
        zone_id=req.zone_id,
        all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": req.detection_type,
            "detected":       True,
            "confidence":     req.confidence,
            "severity":       req.severity,
            "evidence":       req.evidence,
            "action_required": "Alert operator",
        },
        signals={
            "temperature": req.current_temp,
            "humidity":    50.0,
        },
    )

    result = await facility_graph.ainvoke(state)

    orchestrator = result.get("orchestrator_decision", {})
    bot_response = result.get("bot_response", {})

    # broadcast → store.html 경보 오버레이
    await broadcast("safety_alert", {
        "persona_name":   req.persona_name,
        "persona_age":    req.persona_age,
        "zone_id":        req.zone_id,
        "event_id":       req.event_id,
        "severity":       req.severity,
        "detection_type": req.detection_type,
        "evidence":       req.evidence,
        "bot_message":    bot_response.get("message", ""),
        "decision":       orchestrator,
    })

    return {
        "ok":         True,
        "bot_message": bot_response.get("message", ""),
        "decision":   orchestrator,
    }
