"""
인사이트 엔진 시뮬레이션 입력 — 상권 인구통계 프로필 생성.

- aggregate_personas(): Nemotron 형태의 개인 페르소나 → DistrictProfile 집계 (순수 함수)
- SAMPLE_PROFILES   : 오프라인 데모/테스트용 사전 집계 프로필
- load_nemotron_profiles(): nvidia/Nemotron-Personas-Korea 스트리밍 집계 (datasets 필요, 선택)
"""
from data_simulation.personas import (
    aggregate_personas,
    SAMPLE_PROFILES,
    load_nemotron_profiles,
)

__all__ = ["aggregate_personas", "SAMPLE_PROFILES", "load_nemotron_profiles"]
