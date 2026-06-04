"""
LangGraph 기반 무인 매장 멀티에이전트 그래프.

모든 진입은 오케스트레이터 디스패처를 거쳐 전문 에이전트로 분배되며,
전문 에이전트 실행 후에는 오케스트레이터 reconcile 단계에서
크로스존·충돌·이상 정책을 일괄 판단한 뒤 actuator가 부수효과를 실행한다.

라우팅:
  START → orchestrator_dispatch
        ├── trigger_type=="safety"   → safety              ─┐
        ├── trigger_type=="customer" → customer            ─┤
        ├── trigger_type=="report"   → report              ─┼─► orchestrator_reconcile → actuator → END
        └── trigger_type=="insight"  → insight → recommend ─┘
"""
from langgraph.graph import StateGraph, END
from langgraph.constants import START

from llm_module.state import FacilityState
from llm_module.agents.safety_agent import safety_node
from llm_module.agents.customer_agent import customer_node
from llm_module.agents.report_agent import report_node
from llm_module.agents.insight_agent import insight_node
from llm_module.agents.recommendation_agent import recommendation_node
from llm_module.agents.orchestrator import (
    orchestrator_dispatch_node,
    orchestrator_reconcile_node,
)
from llm_module.agents.actuator import actuator_node


def _route_after_dispatch(state: FacilityState) -> str:
    return state["trigger_type"]


def build_graph():
    graph = StateGraph(FacilityState)

    graph.add_node("orchestrator_dispatch", orchestrator_dispatch_node)
    graph.add_node("safety", safety_node)
    graph.add_node("customer", customer_node)
    graph.add_node("report", report_node)
    graph.add_node("insight", insight_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("orchestrator_reconcile", orchestrator_reconcile_node)
    graph.add_node("actuator", actuator_node)

    # 진입: 항상 오케스트레이터 디스패처
    graph.add_edge(START, "orchestrator_dispatch")

    # 디스패처: trigger_type 기반으로 전문 에이전트로 분배
    graph.add_conditional_edges(
        "orchestrator_dispatch",
        _route_after_dispatch,
        {
            "safety": "safety",
            "customer": "customer",
            "report": "report",
            "insight": "insight",
        },
    )

    # 전문 에이전트: 항상 reconcile 단계로
    graph.add_edge("safety", "orchestrator_reconcile")
    graph.add_edge("customer", "orchestrator_reconcile")
    graph.add_edge("report", "orchestrator_reconcile")

    # 인사이트 경로: insight → recommendation → reconcile (부수효과 없이 통과)
    graph.add_edge("insight", "recommendation")
    graph.add_edge("recommendation", "orchestrator_reconcile")

    # Reconcile: 항상 actuator로
    graph.add_edge("orchestrator_reconcile", "actuator")

    # Actuator: 종료
    graph.add_edge("actuator", END)

    return graph.compile()


# 모듈 임포트 시 그래프 빌드 (싱글톤)
facility_graph = build_graph()
