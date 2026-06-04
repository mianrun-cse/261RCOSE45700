"""
인사이트 엔진 end-to-end 데모/검증 스크립트.

상권 인구통계 프로필(DistrictProfile)을 그래프에 넣어
insight → recommendation 노드를 거쳐 자연어 인사이트 + 상품 추천을 출력한다.

실행:
  python test_insight.py                  # SAMPLE_PROFILES 전부
  python test_insight.py 서울-강남구       # 특정 district만
  python test_insight.py --nemotron 서울-강남구   # Nemotron 스트리밍 집계 (datasets 필요)

OPENAI_API_KEY가 필요하다 (.env). 부수효과(알림/온도/TTS)는 없다.
"""
import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from llm_module.graph import facility_graph
from llm_module.state import make_insight_state
from data_simulation import SAMPLE_PROFILES, aggregate_personas, load_nemotron_profiles


def _print_profile(profile: dict) -> None:
    age = profile.get("age", {})
    print(f"  표본 {profile.get('sample_size')}명 / 평균연령 {age.get('mean')} / 연령대 {age.get('cohorts')}")
    occ = ", ".join(o["name"] for o in profile.get("top_occupations", [])[:3])
    print(f"  주요 직업: {occ}")
    print(f"  가구형태: {profile.get('family_types')}")


def _print_result(result: dict) -> None:
    insight = result.get("insight_result") or {}
    rec = result.get("recommendations") or {}

    print(f"\n  ── 인사이트 ──")
    print(f"  요약: {insight.get('summary')}")
    for ins in insight.get("insights", []):
        print(f"   • [{ins.get('category')}|{ins.get('confidence')}] {ins.get('insight')}")
    print(f"  피크 시간대: {insight.get('peak_hours')}")
    print(f"  적합 업종: {insight.get('recommended_store_types')}")

    print(f"\n  ── 상품 추천 ──")
    print(f"  업태 적합도: {rec.get('store_type_fit')}")
    for p in rec.get("products", []):
        print(f"   • {p.get('name')} [{p.get('demand')}] {p.get('peak_window')} "
              f"{p.get('stock_adjustment')} — {p.get('rationale')}")


async def run_profile(profile: dict) -> dict:
    district = profile.get("district")
    print(f"\n{'='*70}\n[{district}]")
    _print_profile(profile)
    result = await facility_graph.ainvoke(make_insight_state(district_profile=profile))
    _print_result(result)
    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("district", nargs="?", help="특정 district만 실행 (생략 시 전체)")
    parser.add_argument("--nemotron", action="store_true",
                        help="Nemotron 스트리밍으로 district 프로필 집계 (datasets 필요)")
    parser.add_argument("--max-rows", type=int, default=50_000,
                        help="--nemotron 스트리밍에서 스캔할 최대 row 수 (기본 50000)")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = parser.parse_args()

    if args.nemotron:
        if not args.district:
            print("--nemotron 사용 시 district를 지정하세요 (예: --nemotron 서울-강남구)")
            sys.exit(1)
        print(f"[Nemotron] '{args.district}' 스트리밍 집계 중... (max_rows={args.max_rows})")
        profiles = load_nemotron_profiles([args.district], max_rows=args.max_rows)
        if not profiles:
            print(f"해당 district 표본을 찾지 못했습니다: {args.district}")
            sys.exit(1)
    else:
        if args.district:
            if args.district not in SAMPLE_PROFILES:
                print(f"알 수 없는 district: {args.district}\n사용 가능: {list(SAMPLE_PROFILES)}")
                sys.exit(1)
            profiles = {args.district: SAMPLE_PROFILES[args.district]}
        else:
            profiles = SAMPLE_PROFILES

    results = {}
    for d, profile in profiles.items():
        results[d] = await run_profile(profile)

    if args.json:
        out = {d: {"insight": r.get("insight_result"), "recommendations": r.get("recommendations")}
               for d, r in results.items()}
        print("\n" + json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
