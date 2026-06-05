"""
서울 열린데이터광장 — 자치구단위 서울 생활인구 일별 집계표
서비스명: SPOP_DAILYSUM_JACHI

특정 자치구 생활인구를 API에서 가져와 SQLite DB에 저장/조회.
"""

import os
import sys
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_KEY   = os.getenv("SEOUL_API_KEY")
BASE_URL  = "http://openapi.seoul.go.kr:8088"
SERVICE   = "SPOP_DAILYSUM_JACHI"
PAGE_SIZE = 1000
DB_PATH   = os.getenv("DB_PATH", "store.db")


# ── DB ───────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS population (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at   TEXT NOT NULL,
                stdr_de_id     TEXT,
                signgu_code    TEXT,
                signgu_nm      TEXT,
                tot_lvpop_co   REAL,
                day_lvpop_co   REAL,
                night_lvpop_co REAL,
                UNIQUE (stdr_de_id, signgu_code)
            )
        """)


def _save(rows: list[dict]):
    if not rows:
        return
    now = datetime.now().isoformat()
    with _conn() as c:
        c.executemany(
            """
            INSERT INTO population
                (collected_at, stdr_de_id, signgu_code, signgu_nm,
                 tot_lvpop_co, day_lvpop_co, night_lvpop_co)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stdr_de_id, signgu_code) DO UPDATE SET
                collected_at   = excluded.collected_at,
                signgu_nm      = excluded.signgu_nm,
                tot_lvpop_co   = excluded.tot_lvpop_co,
                day_lvpop_co   = excluded.day_lvpop_co,
                night_lvpop_co = excluded.night_lvpop_co
            """,
            [
                (
                    now,
                    r.get("STDR_DE_ID"),
                    r.get("SIGNGU_CODE_SE"),
                    r.get("SIGNGU_NM"),
                    _f(r.get("TOT_LVPOP_CO")),
                    _f(r.get("DAY_LVPOP_CO")),
                    _f(r.get("NIGHT_LVPOP_CO")),
                )
                for r in rows
            ],
        )
    print(f"[seoulpublic] {len(rows)}건 저장 → {DB_PATH}")


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── 수집 + 저장 ───────────────────────────────────────────────────

def collect(signgu_code: str, date: str = None) -> list[dict]:
    """
    특정 자치구 생활인구를 API에서 가져와 DB에 저장하고 반환한다.

    Parameters
    ----------
    signgu_code : 시군구코드 (예: "11110" = 종로구)
    date        : 기준일 YYYYMMDD (None이면 전체)
    """
    if not API_KEY:
        raise EnvironmentError("SEOUL_API_KEY가 .env에 설정되지 않았습니다.")

    _init_db()
    all_rows, start = [], 1

    while True:
        end   = start + PAGE_SIZE - 1
        parts = [BASE_URL, API_KEY, "json", SERVICE, str(start), str(end)]
        if date:
            parts.append(date)
        parts.append(signgu_code)

        resp = requests.get("/".join(parts), timeout=10)
        resp.raise_for_status()
        result = resp.json().get(SERVICE, {})
        code   = result.get("RESULT", {}).get("CODE", "")

        if code != "INFO-000":
            print(f"[API 오류] {code}: {result.get('RESULT', {}).get('MESSAGE')}")
            break

        rows  = result.get("row", [])
        total = result.get("list_total_count", 0)
        all_rows.extend(rows)
        print(f"[{start}~{end}] {len(rows)}건 수신 (전체 {total}건)")

        if end >= total:
            break
        start += PAGE_SIZE

    _save(all_rows)
    return all_rows


# ── DB 조회 ───────────────────────────────────────────────────────

def get_population(
    signgu_code: str = None,
    date: str = None,
    limit: int = 100,
) -> list[dict]:
    """DB에서 생활인구 데이터를 조회한다."""
    _init_db()
    clauses, params = [], []
    if signgu_code:
        clauses.append("signgu_code = ?")
        params.append(signgu_code)
    if date:
        clauses.append("stdr_de_id = ?")
        params.append(date)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM population {where} ORDER BY stdr_de_id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "11110"
    date = sys.argv[2] if len(sys.argv) > 2 else None

    collect(signgu_code=code, date=date)

    result = get_population(signgu_code=code, date=date, limit=10)
    print(f"\n[DB 조회 - {len(result)}건]")
    print(f"{'기준일':<12} {'자치구명':<10} {'총생활인구':>12} {'주간인구':>12} {'야간인구':>12}")
    print("-" * 60)
    for r in result:
        print(
            f"{r['stdr_de_id'] or '':<12} "
            f"{r['signgu_nm'] or '':<10} "
            f"{int(r['tot_lvpop_co'] or 0):>12,} "
            f"{int(r['day_lvpop_co'] or 0):>12,} "
            f"{int(r['night_lvpop_co'] or 0):>12,}"
        )
