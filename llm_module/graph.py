"""
LangGraph 기반 무인 매장 멀티에이전트 그래프.

진입점 라우팅:
  safety   → safety_node   → [orchestrator | actuator]
  customer → customer_node → [orchestrator | actuator]
  report   → report_node   → [orchestrator | actuator]
  orchestrator → actuator
  actuator → END

결정 노드(safety/customer/report/orchestrator)는 부수효과를 직접 실행하지 않고
실행 의도를 state["pending_actions"]에 발행하며, actuator 노드가 그래프 말단에서
이를 일괄 실행한다.
"""
from langgraph.graph import StateGraph, END
from langgraph.constants import START

from llm_module.state import FacilityState
from llm_module.agents.safety_agent import safety_node
from llm_module.agents.customer_agent import customer_node
from llm_module.agents.report_agent import report_node
from llm_module.agents.orchestrator import orchestrator_node
from llm_module.agents.actuator import actuator_node


# ── 라우팅 함수 ────────────────────────────────────────────────────────────────

def _route_entry(state: FacilityState) -> str:
    return state["trigger_type"]


def _route_after_safety(state: FacilityState) -> str:
    return "orchestrator" if state.get("conflict_detected") else "actuator"


def _route_after_customer(state: FacilityState) -> str:
    return "orchestrator" if state.get("cross_zone_request") else "actuator"


def _route_after_report(state: FacilityState) -> str:
    return "orchestrator" if state.get("escalate_to_human") or state.get("anomaly_detected") else "actuator"


# ── 그래프 빌드 ────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(FacilityState)

    graph.add_node("safety", safety_node)
    graph.add_node("customer", customer_node)
    graph.add_node("report", report_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("actuator", actuator_node)

    # 진입점 조건 라우팅
    graph.add_conditional_edges(
        START,
        _route_entry,
        {
            "safety": "safety",
            "customer": "customer",
            "report": "report",
        },
    )

    # 안전 에이전트: 충돌 시 오케스트레이터, 아니면 실행 노드
    graph.add_conditional_edges(
        "safety",
        _route_after_safety,
        {"orchestrator": "orchestrator", "actuator": "actuator"},
    )

    # 고객봇 에이전트: 크로스존 요청 시 오케스트레이터, 아니면 실행 노드
    graph.add_conditional_edges(
        "customer",
        _route_after_customer,
        {"orchestrator": "orchestrator", "actuator": "actuator"},
    )

    # 보고서 에이전트: 이상 패턴 감지 시 오케스트레이터, 아니면 실행 노드
    graph.add_conditional_edges(
        "report",
        _route_after_report,
        {"orchestrator": "orchestrator", "actuator": "actuator"},
    )

    # 오케스트레이터: 항상 실행 노드로
    graph.add_edge("orchestrator", "actuator")

    # 실행 노드: 항상 종료
    graph.add_edge("actuator", END)

    return graph.compile()


# 모듈 임포트 시 그래프 빌드 (싱글톤)
facility_graph = build_graph()
