"""
상권 인구통계 프로필(DistrictProfile) 생성 — 인사이트 엔진의 입력 공급.

데이터 출처: nvidia/Nemotron-Personas-Korea (개인 단위 합성 페르소나 1M건).
이 모듈은 개인 페르소나를 district 기준으로 집계해 DistrictProfile로 변환한다.

격리 원칙: 에이전트(insight/recommendation)는 DistrictProfile dict에만 의존한다.
Lee의 docs/schema.md가 확정되면 _NEMOTRON_FIELDS 매핑과 aggregate_personas()만
손보면 되고, 에이전트 코드는 변경되지 않는다.
"""
from __future__ import annotations

import ast
from collections import Counter
from typing import Iterable

from llm_module.insight_schema import DistrictProfile

# Nemotron 원본 → DistrictProfile에 사용하는 필드명. (스키마 확정 시 여기만 수정)
_NEMOTRON_FIELDS = {
    "district": "district",
    "province": "province",
    "age": "age",
    "sex": "sex",
    "occupation": "occupation",
    "family_type": "family_type",
    "housing_type": "housing_type",
    "education_level": "education_level",
    "hobbies_list": "hobbies_and_interests_list",
}

# 연령 코호트 경계 (하한 포함, 상한 미만). team_pivot 데모용 5구간.
_AGE_COHORTS = [
    ("19-29", 19, 30),
    ("30-39", 30, 40),
    ("40-49", 40, 50),
    ("50-64", 50, 65),
    ("65+", 65, 200),
]


def _round_ratios(counter: Counter, total: int, top_n: int | None = None) -> dict[str, float]:
    """Counter → {라벨: 비율} (소수 셋째 자리 반올림). top_n 지정 시 상위 N개만."""
    if total <= 0:
        return {}
    items = counter.most_common(top_n) if top_n else counter.most_common()
    return {label: round(cnt / total, 3) for label, cnt in items}


def _parse_hobbies(raw) -> list[str]:
    """hobbies_and_interests_list는 "['a', 'b']" 형태의 문자열 → 실제 리스트로 파싱."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def aggregate_personas(
    rows: Iterable[dict],
    *,
    top_occupations: int = 6,
    top_family_types: int = 5,
    top_housing_types: int = 5,
    top_education: int = 5,
    top_hobbies: int = 8,
) -> DistrictProfile | None:
    """
    동일 district에 속한 개인 페르소나 row들을 DistrictProfile로 집계.

    rows: Nemotron 형태의 dict 이터러블. 모두 같은 district라고 가정한다
          (서로 다른 district가 섞여 있으면 load_nemotron_profiles가 분리 호출).
    반환: DistrictProfile, 입력이 비면 None.
    """
    F = _NEMOTRON_FIELDS
    rows = list(rows)
    if not rows:
        return None

    n = len(rows)
    first = rows[0]

    ages: list[int] = []
    sex = Counter()
    occ = Counter()
    fam = Counter()
    house = Counter()
    edu = Counter()
    hobby = Counter()

    for r in rows:
        age_val = r.get(F["age"])
        if isinstance(age_val, (int, float)):
            ages.append(int(age_val))
        if r.get(F["sex"]):
            sex[r[F["sex"]]] += 1
        if r.get(F["occupation"]):
            occ[r[F["occupation"]]] += 1
        if r.get(F["family_type"]):
            fam[r[F["family_type"]]] += 1
        if r.get(F["housing_type"]):
            house[r[F["housing_type"]]] += 1
        if r.get(F["education_level"]):
            edu[r[F["education_level"]]] += 1
        for h in _parse_hobbies(r.get(F["hobbies_list"])):
            hobby[h] += 1

    # 연령 코호트 비율
    cohort_counts = Counter()
    for a in ages:
        for label, lo, hi in _AGE_COHORTS:
            if lo <= a < hi:
                cohort_counts[label] += 1
                break
    cohorts = {
        label: round(cohort_counts.get(label, 0) / len(ages), 3)
        for label, _, _ in _AGE_COHORTS
    } if ages else {}

    profile: DistrictProfile = {
        "district": first.get(F["district"], "unknown"),
        "province": first.get(F["province"], "unknown"),
        "sample_size": n,
        "age": {
            "mean": round(sum(ages) / len(ages), 1) if ages else 0.0,
            "cohorts": cohorts,
        },
        "sex_ratio": _round_ratios(sex, n),
        "top_occupations": [
            {"name": name, "ratio": round(cnt / n, 3)}
            for name, cnt in occ.most_common(top_occupations)
        ],
        "family_types": _round_ratios(fam, n, top_family_types),
        "housing_types": _round_ratios(house, n, top_housing_types),
        "education_levels": _round_ratios(edu, n, top_education),
        "top_hobbies": [h for h, _ in hobby.most_common(top_hobbies)],
    }
    return profile


def load_nemotron_profiles(
    districts: list[str] | None = None,
    *,
    max_rows: int = 50_000,
) -> dict[str, DistrictProfile]:
    """
    nvidia/Nemotron-Personas-Korea를 스트리밍으로 읽어 district별 DistrictProfile 생성.

    `datasets` 패키지가 필요하다 (pip install datasets). 미설치 시 안내 후 예외.
    districts: 집계할 district 화이트리스트(예: ["서울-강남구"]). None이면 전부.
    max_rows : 스트리밍에서 읽을 최대 row 수(데모용 상한; 1M 전체는 비현실적).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "load_nemotron_profiles는 datasets 패키지가 필요합니다. "
            "`pip install datasets` 후 다시 시도하거나, 오프라인이면 SAMPLE_PROFILES를 사용하세요."
        ) from e

    ds = load_dataset(
        "nvidia/Nemotron-Personas-Korea", split="train", streaming=True
    )
    wanted = set(districts) if districts else None
    buckets: dict[str, list[dict]] = {}

    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        d = row.get(_NEMOTRON_FIELDS["district"])
        if not d or (wanted is not None and d not in wanted):
            continue
        buckets.setdefault(d, []).append(row)

    return {
        d: prof
        for d, rows in buckets.items()
        if (prof := aggregate_personas(rows)) is not None
    }


# ── 오프라인 데모/테스트용 사전 집계 프로필 ──────────────────────────────────
# 실제 Nemotron 분포를 손으로 근사한 샘플. 인터넷/대용량 다운로드 없이 즉시 동작한다.
SAMPLE_PROFILES: dict[str, DistrictProfile] = {
    "서울-강남구": {
        "district": "서울-강남구",
        "province": "서울",
        "sample_size": 1840,
        "age": {
            "mean": 41.2,
            "cohorts": {"19-29": 0.18, "30-39": 0.27, "40-49": 0.22, "50-64": 0.23, "65+": 0.10},
        },
        "sex_ratio": {"남자": 0.48, "여자": 0.52},
        "top_occupations": [
            {"name": "경영 및 진단 전문가", "ratio": 0.09},
            {"name": "회계 사무원", "ratio": 0.07},
            {"name": "소프트웨어 개발자", "ratio": 0.06},
            {"name": "의사", "ratio": 0.04},
            {"name": "변호사", "ratio": 0.03},
            {"name": "마케팅 전문가", "ratio": 0.03},
        ],
        "family_types": {"1인 가구": 0.36, "배우자와 거주": 0.24, "배우자·자녀와 거주": 0.22, "부모와 거주": 0.10},
        "housing_types": {"아파트": 0.61, "오피스텔": 0.18, "다세대주택": 0.12, "단독주택": 0.06},
        "education_levels": {"4년제 대학교": 0.48, "대학원": 0.21, "2~3년제 전문대학": 0.15, "고등학교": 0.13},
        "top_hobbies": ["헬스장 운동", "와인 모임", "전시회 관람", "브런치 카페 탐방", "주말 골프", "러닝"],
    },
    "광주-서구": {
        "district": "광주-서구",
        "province": "광주",
        "sample_size": 920,
        "age": {
            "mean": 49.8,
            "cohorts": {"19-29": 0.12, "30-39": 0.17, "40-49": 0.20, "50-64": 0.30, "65+": 0.21},
        },
        "sex_ratio": {"남자": 0.50, "여자": 0.50},
        "top_occupations": [
            {"name": "하역 및 적재 관련 단순 종사원", "ratio": 0.08},
            {"name": "자영업자", "ratio": 0.07},
            {"name": "건설 단순 종사원", "ratio": 0.05},
            {"name": "음식점 종사원", "ratio": 0.05},
            {"name": "운수 종사원", "ratio": 0.04},
        ],
        "family_types": {"배우자·자녀와 거주": 0.31, "배우자와 거주": 0.27, "1인 가구": 0.21, "3세대 거주": 0.08},
        "housing_types": {"아파트": 0.52, "단독주택": 0.21, "다세대주택": 0.18, "연립주택": 0.07},
        "education_levels": {"고등학교": 0.39, "4년제 대학교": 0.24, "중학교": 0.16, "2~3년제 전문대학": 0.14},
        "top_hobbies": ["무등산 둘레길 산책", "전통시장 맛집 탐방", "트로트 프로그램 시청", "동네 대중사우나 이용", "텃밭 가꾸기"],
    },
}
