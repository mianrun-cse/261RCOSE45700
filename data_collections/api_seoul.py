"""
서울시 실시간 도시데이터 API (OA-21285)
한 번 호출로 특정 지역의 인구·날씨·교통 정보를 모두 수집해 store.db에 저장한다.

수집 feature 요약
─────────────────────────────────────────────────────────────
[인구] 실시간 유동인구 규모 및 구성
  area_congest_lvl    혼잡도 레벨 (여유/보통/약간붐빔/붐빔)
  area_ppltn_min/max  실시간 인구 추정 범위
  ppltn_rate_10~70    연령대별 비율 (10대~70대 이상)
  resnt_ppltn_rate    거주인구 비율
  non_resnt_ppltn_rate 비거주인구 비율

[날씨] 강수 여부 및 방문 수요에 영향을 주는 feature
  precpt_type         강수형태 (없음/비/눈/비+눈)  ← 핵심
  precipitation       강수량 (mm)
  temp                기온
  sensible_temp       체감온도
  humidity            습도
  wind_spd            풍속
  sky_stts            하늘 상태 (맑음/구름많음/흐림)
  pm10                미세먼지 농도
  air_idx             통합대기환경지수

[교통] 사람이 얼마나 이동했는지
  road_traffic_idx    도로 소통 현황 (원활/서행/지체/정체)
  road_traffic_spd    도로 평균 속도 (km/h)
─────────────────────────────────────────────────────────────
"""

import os
import sys
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

API_KEY  = os.getenv("SEOUL_API_KEY")
BASE_URL = "http://openapi.seoul.go.kr:8088"
DB_PATH  = os.getenv("DB_PATH", "store.db")


# ── DB ────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS city_data (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at         TEXT NOT NULL,
                area_nm              TEXT NOT NULL,

                -- 인구
                area_congest_lvl     TEXT,
                area_ppltn_min       INTEGER,
                area_ppltn_max       INTEGER,
                ppltn_rate_10        REAL,
                ppltn_rate_20        REAL,
                ppltn_rate_30        REAL,
                ppltn_rate_40        REAL,
                ppltn_rate_50        REAL,
                ppltn_rate_60        REAL,
                ppltn_rate_70        REAL,
                resnt_ppltn_rate     REAL,
                non_resnt_ppltn_rate REAL,

                -- 날씨
                precpt_type          TEXT,
                precipitation        REAL,
                temp                 REAL,
                sensible_temp        REAL,
                humidity             REAL,
                wind_spd             REAL,
                sky_stts             TEXT,
                pm10                 REAL,
                air_idx              TEXT,

                -- 교통
                road_traffic_idx     TEXT,
                road_traffic_spd     REAL
            )
        """)


def _save(area_nm: str, row: dict):
    now = datetime.now().isoformat()
    with _conn() as c:
        c.execute("""
            INSERT INTO city_data (
                collected_at, area_nm,
                area_congest_lvl, area_ppltn_min, area_ppltn_max,
                ppltn_rate_10, ppltn_rate_20, ppltn_rate_30,
                ppltn_rate_40, ppltn_rate_50, ppltn_rate_60, ppltn_rate_70,
                resnt_ppltn_rate, non_resnt_ppltn_rate,
                precpt_type, precipitation, temp, sensible_temp,
                humidity, wind_spd, sky_stts, pm10, air_idx,
                road_traffic_idx, road_traffic_spd
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            now, area_nm,
            row.get("area_congest_lvl"), row.get("area_ppltn_min"), row.get("area_ppltn_max"),
            row.get("ppltn_rate_10"), row.get("ppltn_rate_20"), row.get("ppltn_rate_30"),
            row.get("ppltn_rate_40"), row.get("ppltn_rate_50"), row.get("ppltn_rate_60"), row.get("ppltn_rate_70"),
            row.get("resnt_ppltn_rate"), row.get("non_resnt_ppltn_rate"),
            row.get("precpt_type"), row.get("precipitation"), row.get("temp"), row.get("sensible_temp"),
            row.get("humidity"), row.get("wind_spd"), row.get("sky_stts"), row.get("pm10"), row.get("air_idx"),
            row.get("road_traffic_idx"), row.get("road_traffic_spd"),
        ))
    print(f"[api_seoul] {area_nm} 저장 완료 → {DB_PATH}")


# ── 파싱 ──────────────────────────────────────────────────────────

def _parse(data: dict) -> dict:
    """API 응답에서 필요한 feature만 추출"""
    result = {}

    # 인구
    ppltn = data.get("LIVE_PPLTN_STTS", [{}])
    if isinstance(ppltn, list):
        ppltn = ppltn[0] if ppltn else {}
    result["area_congest_lvl"]     = ppltn.get("AREA_CONGEST_LVL")
    result["area_ppltn_min"]       = _i(ppltn.get("AREA_PPLTN_MIN"))
    result["area_ppltn_max"]       = _i(ppltn.get("AREA_PPLTN_MAX"))
    result["ppltn_rate_10"]        = _f(ppltn.get("PPLTN_RATE_10"))
    result["ppltn_rate_20"]        = _f(ppltn.get("PPLTN_RATE_20"))
    result["ppltn_rate_30"]        = _f(ppltn.get("PPLTN_RATE_30"))
    result["ppltn_rate_40"]        = _f(ppltn.get("PPLTN_RATE_40"))
    result["ppltn_rate_50"]        = _f(ppltn.get("PPLTN_RATE_50"))
    result["ppltn_rate_60"]        = _f(ppltn.get("PPLTN_RATE_60"))
    result["ppltn_rate_70"]        = _f(ppltn.get("PPLTN_RATE_70"))
    result["resnt_ppltn_rate"]     = _f(ppltn.get("RESNT_PPLTN_RATE"))
    result["non_resnt_ppltn_rate"] = _f(ppltn.get("NON_RESNT_PPLTN_RATE"))

    # 날씨
    weather = data.get("WEATHER_STTS", [{}])
    if isinstance(weather, list):
        weather = weather[0] if weather else {}
    result["precpt_type"]   = weather.get("PRECPT_TYPE")
    result["precipitation"] = _f(weather.get("PRECIPITATION"))
    result["temp"]          = _f(weather.get("TEMP"))
    result["sensible_temp"] = _f(weather.get("SENSIBLE_TEMP"))
    result["humidity"]      = _f(weather.get("HUMIDITY"))
    result["wind_spd"]      = _f(weather.get("WIND_SPD"))
    result["sky_stts"]      = weather.get("SKY_STTS")
    result["pm10"]          = _f(weather.get("PM10"))
    result["air_idx"]       = weather.get("AIR_IDX")

    # 교통 — 평균값은 AVG_ROAD_DATA 하위에 있음
    traffic_stts = data.get("ROAD_TRAFFIC_STTS", {})
    if isinstance(traffic_stts, list):
        traffic_stts = traffic_stts[0] if traffic_stts else {}
    avg = traffic_stts.get("AVG_ROAD_DATA", {})
    result["road_traffic_idx"] = avg.get("ROAD_TRAFFIC_IDX")
    result["road_traffic_spd"] = _f(avg.get("ROAD_TRAFFIC_SPD"))

    return result


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ── 수집 ──────────────────────────────────────────────────────────

def collect(area_nm: str) -> dict:
    """
    특정 지역의 인구·날씨·교통 데이터를 수집해 DB에 저장한다.

    Parameters
    ----------
    area_nm : 지역명 (예: "강남 MICE 관광특구", "광화문·덕수궁")

    Returns
    -------
    수집된 feature dict
    """
    if not API_KEY:
        raise EnvironmentError("SEOUL_API_KEY가 .env에 설정되지 않았습니다.")

    url  = f"{BASE_URL}/{API_KEY}/json/citydata/1/5/{requests.utils.quote(area_nm)}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    body = resp.json().get("CITYDATA", {})

    _init_db()
    row = _parse(body)
    _save(area_nm, row)
    return row


# ── 조회 ──────────────────────────────────────────────────────────

def get_data(area_nm: str = None, limit: int = 50) -> list[dict]:
    """DB에서 수집된 데이터를 조회한다."""
    _init_db()
    params = []
    where  = ""
    if area_nm:
        where = "WHERE area_nm = ?"
        params.append(area_nm)
    params.append(limit)

    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM city_data {where} ORDER BY collected_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    area = sys.argv[1] if len(sys.argv) > 1 else "광화문·덕수궁"

    row = collect(area)

    print(f"\n{'=' * 50}")
    print(f"지역: {area}")
    print(f"{'=' * 50}")
    def _d(v, unit=""):
        return f"{v}{unit}" if v is not None else "-"

    print(f"[인구]")
    print(f"  혼잡도     : {row.get('area_congest_lvl') or '-'}")
    print(f"  인구 범위  : {row.get('area_ppltn_min') or 0:,} ~ {row.get('area_ppltn_max') or 0:,}명")
    print(f"  거주인구   : {_d(row.get('resnt_ppltn_rate'), '%')}  비거주: {_d(row.get('non_resnt_ppltn_rate'), '%')}")
    print(f"\n[날씨]")
    print(f"  강수형태   : {row.get('precpt_type') or '-'}  강수량: {_d(row.get('precipitation'), 'mm')}")
    print(f"  기온       : {_d(row.get('temp'), '°C')}  체감: {_d(row.get('sensible_temp'), '°C')}")
    print(f"  습도       : {_d(row.get('humidity'), '%')}  풍속: {_d(row.get('wind_spd'), 'm/s')}")
    print(f"  하늘상태   : {row.get('sky_stts') or '-'}  미세먼지: {_d(row.get('pm10'), '㎍/㎥')}")
    print(f"\n[교통]")
    print(f"  도로 소통  : {row.get('road_traffic_idx') or '-'}")
    print(f"  평균 속도  : {_d(row.get('road_traffic_spd'), 'km/h')}")
