"""
인사이트 엔진 입력 계약 — 상권 인구통계 프로필 (DistrictProfile).

이 스키마는 nvidia/Nemotron-Personas-Korea 데이터셋의 개인 단위 페르소나를
`district`(예: "서울-강남구") 기준으로 집계한 결과를 표현한다.
인사이트/추천 에이전트의 단일 입력 계약이며, 집계 로직은
`data_simulation/personas.py`의 aggregate_personas()에 격리되어 있다.

※ 임시 계약: Lee의 docs/schema.md가 확정되면 aggregate_personas() 한 곳의
  필드 매핑만 수정하면 된다. 에이전트 코드는 이 dict 형태에만 의존한다.

Nemotron 원본 필드 → DistrictProfile 매핑:
  district          → district          (그룹 키)
  province          → province
  age (int)         → age.mean / age.cohorts
  sex               → sex_ratio
  occupation        → top_occupations
  family_type       → family_types
  housing_type      → housing_types
  education_level   → education_levels
  hobbies_*_list    → top_hobbies
  (없음)            → hourly_footfall  ← 교통데이터 담당이 추후 공급, 없으면 LLM 추정
"""
from typing import TypedDict, List, Dict, Optional


class AgeProfile(TypedDict, total=False):
    mean: float
    cohorts: Dict[str, float]   # {"19-29": 0.18, "30-39": 0.27, ...} 비율 합 ≈ 1.0


class OccupationShare(TypedDict):
    name: str
    ratio: float


class DistrictProfile(TypedDict, total=False):
    # 식별 (필수)
    district: str                       # "서울-강남구"  (Nemotron district)
    province: str                       # "서울"
    sample_size: int                    # 집계에 사용된 페르소나 수

    # 인구통계 분포
    age: AgeProfile
    sex_ratio: Dict[str, float]         # {"남자": 0.48, "여자": 0.52}
    top_occupations: List[OccupationShare]
    family_types: Dict[str, float]      # {"1인 가구": 0.34, ...}
    housing_types: Dict[str, float]     # {"아파트": 0.55, ...}
    education_levels: Dict[str, float]  # {"4년제 대학교": 0.41, ...}
    top_hobbies: List[str]              # ["고궁 산책", "트로트 시청", ...]

    # 선택 — Nemotron에 없는 시간대별 유동인구(교통데이터 담당 공급). 없으면 LLM이 인구통계로 추정.
    hourly_footfall: Optional[Dict[str, float]]  # {"07": 0.10, "08": 0.18, ...}


# ── 인사이트 엔진 출력 계약 (참고용 문서화) ───────────────────────────────────
# insight_node 출력  → state["insight_result"]:
#   {
#     "summary": str,                                   # 상권 한 줄 요약
#     "insights": [{"category": str, "insight": str, "confidence": "high|medium|low"}],
#     "peak_hours": [str],                              # ["07:00-09:00", "22:00-01:00"]
#     "recommended_store_types": [str],                 # ["고단백 스낵 자판기", "무인 세탁소"]
#   }
#
# recommendation_node 출력 → state["recommendations"]:
#   {
#     "location": str,
#     "products": [{"name": str, "demand": "high|moderate|steady|low",
#                   "peak_window": str, "stock_adjustment": str, "rationale": str}],
#     "store_type_fit": str,
#   }
