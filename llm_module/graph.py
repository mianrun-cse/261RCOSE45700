"""
LangGraph 기반 시설 멀티에이전트 그래프.

진입점 라우팅:
  safety   → safety_node  → [orchestrator | END]
  customer → customer_node → [orchestrator | END]
  coaching → coaching_node → END
  report   → report_node  → [orchestrator | END]
  orchestrator → END
"""
from langgraph.graph import StateGraph, END
from langgraph.constants import START

from llm_module.state import FacilityState
from llm_module.agents.safety_agent import safety_node
from llm_module.agents.customer_agent import customer_node
from llm_module.agents.coaching_agent import coaching_node
from llm_module.agents.report_agent import report_node
from llm_module.agents.orchestrator import orchestrator_node


# ── 라우팅 함수 ────────────────────────────────────────────────────────────────

def _route_entry(state: FacilityState) -> str:
    return state["trigger_type"]


def _route_after_safety(state: FacilityState) -> str:
    return "orchestrator" if state.get("conflict_detected") else END


def _route_after_customer(state: FacilityState) -> str:
    return "orchestrator" if state.get("cross_bay_request") else END


def _route_after_report(state: FacilityState) -> str:
    return "orchestrator" if state.get("escalate_to_human") or state.get("anomaly_detected") else END


# ── 그래프 빌드 ────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(FacilityState)

    graph.add_node("safety", safety_node)
    graph.add_node("customer", customer_node)
    graph.add_node("coaching", coaching_node)
    graph.add_node("report", report_node)
    graph.add_node("orchestrator", orchestrator_node)

    # 진입점 조건 라우팅
    graph.add_conditional_edges(
        START,
        _route_entry,
        {
            "safety": "safety",
            "customer": "customer",
            "coaching": "coaching",
            "report": "report",
        },
    )

    # 안전 에이전트: 충돌 시 오케스트레이터, 아니면 종료
    graph.add_conditional_edges(
        "safety",
        _route_after_safety,
        {"orchestrator": "orchestrator", END: END},
    )

    # 고객봇 에이전트: 크로스베이 요청 시 오케스트레이터, 아니면 종료
    graph.add_conditional_edges(
        "customer",
        _route_after_customer,
        {"orchestrator": "orchestrator", END: END},
    )

    # 코칭 에이전트: 항상 종료
    graph.add_edge("coaching", END)

    # 보고서 에이전트: 이상 패턴 감지 시 오케스트레이터, 아니면 종료
    graph.add_conditional_edges(
        "report",
        _route_after_report,
        {"orchestrator": "orchestrator", END: END},
    )

    # 오케스트레이터: 항상 종료
    graph.add_edge("orchestrator", END)

    return graph.compile()


# 모듈 임포트 시 그래프 빌드 (싱글톤)
facility_graph = build_graph()
