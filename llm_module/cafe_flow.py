import asyncio
import json
import os
import re
from typing import Any

from openai import OpenAI

client = OpenAI()

AVAILABLE_ZONES = [
    "entrance", "kiosk", "machine", "pickup",
    "icemachine", "dessert",
    "window_seat", "study_zone",
    "table_2a", "table_2b", "table_2c", "table_2d",
    "sofa_lounge", "group_table",
    "trash", "bathroom",
]

SYSTEM_PROMPT = """
You are a retail behavior analyst specializing in unmanned (무인) cafe customer flows.

Given a district demographic profile, generate a realistic JSON configuration
for a cafe customer simulation.

Rules:
1. flow_stages must be ordered as the customer journey
   — always start with "kiosk" (prob 1.0) and end with "exit" (prob 1.0)
   — "sitting" stage MUST have zone set to null (assigned dynamically)
   — intermediate stages (waiting, dessert browsing, etc.) can have probability < 1.0
   — duration_range is in simulation seconds [min, max]
   — each stage MUST include an "actions" array: 3-5 Korean strings with emojis
   — each stage MUST use field names: "id", "probability", "duration_range", "zone", "actions", "label"
2. drink_menu: 8-12 items realistic for Korean unmanned cafe
3. dessert_menu: 4-6 items
4. seat_preferences: weights for under_30 / 30s / 40s / 50plus
   — zone ids must be from the available list
5. spawn_rate: seconds between customer spawns per time period
   — reflect district peak hours (office district = busy lunch, residential = busy morning)
6. persona_types: 3-5 types reflecting the district demographics
   — weight must sum to 1.0
   — preferred_activities: 3-5 Korean action strings with emoji
7. All Korean strings. Use emojis liberally in action labels.
8. Return ONLY valid JSON. No markdown, no explanation.
9. IMPORTANT: Use exact field names — "id" not "stage", "probability" not "prob"

Available zone IDs: """ + ", ".join(AVAILABLE_ZONES)


# ─────────────────────────────────────────────
#  Comfort / Safety event config
# ─────────────────────────────────────────────

# How often (sim-seconds) to roll for a comfort/safety event per persona
COMFORT_CHECK_INTERVAL = 30.0   # every 30 sim-seconds while sitting
SAFETY_CHECK_INTERVAL  = 60.0   # every 60 sim-seconds anywhere

# Probability per check
COMFORT_EVENT_PROB = 0.15   # 15% chance persona feels uncomfortable
SAFETY_EVENT_PROB  = 0.05   # 5%  chance persona feels unsafe

COMFORT_EVENTS = [
    {
        "id":      "too_hot",
        "message": "너무 더워요, 에어컨 좀 낮춰주세요 🥵",
        "action":  {"type": "temperature", "value": -2},   # delta °C
        "bubble":  "🥵 너무 더워요!",
        "log_class": "warn",
        "log_text":  "온도 불편 호소",
    },
    {
        "id":      "too_cold",
        "message": "너무 추워요, 에어컨 좀 올려주세요 🥶",
        "action":  {"type": "temperature", "value": +2},
        "bubble":  "🥶 너무 추워요!",
        "log_class": "warn",
        "log_text":  "냉기 불편 호소",
    },
    {
        "id":      "stuffy",
        "message": "환기가 안 되는 것 같아요, 팬 좀 켜주세요 😮‍💨",
        "action":  {"type": "fan", "value": "on"},
        "bubble":  "😮‍💨 답답해요!",
        "log_class": "warn",
        "log_text":  "환기 불편 호소",
    },
]

SAFETY_EVENTS = [
    {
        "id":        "suspicious_person",
        "detection_type": "suspicious_behavior",
        "confidence": 0.82,
        "severity":   "medium",
        "evidence":   "Customer reported feeling unsafe near another patron",
        "bubble":     "😨 무서워요!",
        "log_class":  "warn",
        "log_text":   "불안 신고",
    },
    {
        "id":        "harassment",
        "detection_type": "harassment",
        "confidence": 0.91,
        "severity":   "high",
        "evidence":   "Customer reported harassment",
        "bubble":     "🚨 도움이 필요해요!",
        "log_class":  "warn",
        "log_text":   "harassment 신고",
    },
]


# ─────────────────────────────────────────────
#  Helper: normalize LLM field names
# ─────────────────────────────────────────────

def _normalize_stage(stage: dict) -> dict:
    """LLM이 잘못된 필드명을 반환할 경우 정규화."""
    if "stage" in stage and "id" not in stage:
        stage["id"] = stage.pop("stage")
    if "prob" in stage and "probability" not in stage:
        stage["probability"] = stage.pop("prob")
    if "duration" in stage and "duration_range" not in stage:
        stage["duration_range"] = stage.pop("duration")
    if "label" not in stage and "id" in stage:
        stage["label"] = stage["id"]
    return stage


# ─────────────────────────────────────────────
#  Helper: default actions per stage id
# ─────────────────────────────────────────────

def _default_actions_for_stage(stage_id: str) -> list[str]:
    """stage id별 의미 있는 기본 액션 반환."""
    defaults = {
        "kiosk":          ["🖥️ 메뉴 보는 중", "🤔 뭐 마실까...", "💳 결제 중"],
        "waiting":        ["⏳ 음료 기다리는 중", "📱 폰 보며 대기", "🍰 디저트 구경"],
        "drink_browsing": ["🥤 음료 구경 중", "👀 메뉴판 보는 중", "🤔 고민 중"],
        "pickup":         ["☕ 음료 받는 중", "🥤 음료 픽업", "😊 음료 확인 중"],
        "sitting":        ["💻 공부 중", "📖 독서 중", "💬 대화 중", "📱 영상 보는 중"],
        "dessert":        ["🍰 디저트 고르는 중", "😋 디저트 구경", "🤤 메뉴 고민 중"],
        "trash":          ["🗑️ 컵 반납 중", "🧹 자리 정리", "♻️ 분리수거 중"],
        "exit":           ["🚶 퇴장 중", "👋 퇴장", "😊 만족스러운 퇴장"],
        "bathroom":       ["🚻 화장실 이용 중"],
        "icemachine":     ["🧊 얼음 채우는 중"],
    }
    return defaults.get(stage_id, ["👀 둘러보는 중"])


# ─────────────────────────────────────────────
#  Default fallbacks
# ─────────────────────────────────────────────

def _default_flow_stages() -> list[dict]:
    return [
        {
            "id": "kiosk", "zone": "kiosk", "label": "키오스크 주문",
            "actions": ["🖥️ 메뉴 보는 중", "🤔 뭐 마실까...", "💳 결제 중"],
            "duration_range": [8, 20], "probability": 1.0,
        },
        {
            "id": "waiting", "zone": "pickup", "label": "음료 대기",
            "actions": ["⏳ 음료 기다리는 중", "📱 폰 보며 대기", "🍰 디저트 구경"],
            "duration_range": [5, 15], "probability": 1.0,
        },
        {
            "id": "pickup", "zone": "pickup", "label": "음료 픽업",
            "actions": ["☕ 음료 받는 중", "🥤 음료 픽업"],
            "duration_range": [2, 5], "probability": 1.0,
        },
        {
            "id": "sitting", "zone": None, "label": "착석",
            "actions": ["💻 공부 중", "📖 독서 중", "💬 대화 중", "📱 영상 보는 중"],
            "duration_range": [30, 120], "probability": 0.85,
        },
        {
            "id": "trash", "zone": "trash", "label": "컵 반납",
            "actions": ["🗑️ 컵 반납 중", "🧹 자리 정리"],
            "duration_range": [3, 6], "probability": 0.9,
        },
        {
            "id": "exit", "zone": "entrance", "label": "퇴장",
            "actions": ["🚶 퇴장 중"],
            "duration_range": [1, 3], "probability": 1.0,
        },
    ]


def _default_seat_prefs() -> dict:
    return {
        "under_30": {"study_zone": 4, "window_seat": 3, "sofa_lounge": 2,
                     "table_2a": 1,   "table_2b": 1},
        "30s":      {"table_2a": 3,   "table_2b": 3,   "window_seat": 2,
                     "study_zone": 2, "sofa_lounge": 2},
        "40s":      {"sofa_lounge": 4, "table_2a": 3,  "table_2b": 3,
                     "window_seat": 1, "group_table": 2},
        "50plus":   {"sofa_lounge": 5, "group_table": 3, "table_2a": 2,
                     "table_2b": 2},
    }


def _default_persona_types() -> list[dict]:
    return [
        {
            "label": "대학생", "emoji_pool": ["🐱", "🐰", "🦊"],
            "age_range": [19, 26], "weight": 0.35,
            "preferred_activities": ["💻 공부 중", "🎧 음악 감상", "🤳 SNS 중"],
        },
        {
            "label": "직장인", "emoji_pool": ["🐨", "🐼", "🦝"],
            "age_range": [27, 45], "weight": 0.45,
            "preferred_activities": ["✏️ 업무 중", "☕ 커피 즐기는 중", "💬 대화 중"],
        },
        {
            "label": "중장년", "emoji_pool": ["🐻", "🦁", "🐯"],
            "age_range": [46, 70], "weight": 0.20,
            "preferred_activities": ["📖 독서 중", "😴 잠깐 휴식", "☕ 커피 즐기는 중"],
        },
    ]


# ─────────────────────────────────────────────
#  Patch defaults
# ─────────────────────────────────────────────

def _patch_defaults(data: dict) -> dict:
    """필수 키가 없으면 안전한 기본값으로 채운다."""

    if "flow_stages" not in data or not data["flow_stages"]:
        data["flow_stages"] = _default_flow_stages()

    if "drink_menu" not in data or len(data["drink_menu"]) < 3:
        data["drink_menu"] = [
            "아메리카노", "카페라떼", "카푸치노", "바닐라라떼",
            "아이스티",   "초코라떼", "그린티라떼", "에스프레소",
        ]

    if "dessert_menu" not in data or not data["dessert_menu"]:
        data["dessert_menu"] = ["크로와상", "머핀", "마카롱", "치즈케이크"]

    if "seat_preferences" not in data:
        data["seat_preferences"] = _default_seat_prefs()

    if "spawn_rate" not in data:
        data["spawn_rate"] = {"morning": 8, "afternoon": 14, "evening": 7}

    if "max_capacity" not in data:
        data["max_capacity"] = 14

    if "persona_types" not in data or not data["persona_types"]:
        data["persona_types"] = _default_persona_types()

    # ✅ Per-stage: normalize FIRST, then fill defaults
    for i, stage in enumerate(data["flow_stages"]):
        stage = _normalize_stage(stage)           # ✅ fix field names first
        data["flow_stages"][i] = stage            # ✅ write back to list

        stage.setdefault("probability",    1.0)
        stage.setdefault("duration_range", [5, 15])
        stage.setdefault("zone",           None)
        stage.setdefault("label",          stage.get("id", "unknown"))

        # ✅ Only fallback if actions truly missing or empty
        if not stage.get("actions"):
            stage["actions"] = _default_actions_for_stage(stage.get("id", ""))

        # Fix string "null" → real None
        if stage.get("zone") in ("null", "None", ""):
            stage["zone"] = None

        # sitting zone must always be None (assigned dynamically)
        if stage.get("id") == "sitting":
            stage["zone"] = None

    # Normalize persona weights to sum = 1.0
    types = data["persona_types"]
    total = sum(t.get("weight", 1) for t in types)
    if total > 0:
        for t in types:
            t["weight"] = round(t.get("weight", 1) / total, 4)

    return data


# ─────────────────────────────────────────────
#  District-aware event configuration
# ─────────────────────────────────────────────

def _compute_event_config(profile: dict) -> dict:
    """
    상권 인구통계(DistrictProfile)에서 이벤트 가중치와 확률을 계산한다.

    핵심 규칙:
      - 고령 비율(50+) 높을수록 : 낙상/열탈진 위험 ↑, 와이파이/충전 불편 ↓
      - 청년 비율(19-39) 높을수록: 와이파이·충전·소음·기물파손 ↑, 낙상 ↓
      - 기술직/학생 비율 높을수록: 어두움·와이파이·충전 ↑↑
    """
    age_info     = profile.get("age", {})
    cohorts      = age_info.get("cohorts", {})
    senior_ratio = cohorts.get("65+", 0.10) + cohorts.get("50-64", 0.20)
    youth_ratio  = cohorts.get("19-29", 0.20) + cohorts.get("30-39", 0.20)

    occ_names = " ".join(o.get("name", "") for o in profile.get("top_occupations", []))
    is_tech   = any(w in occ_names for w in ["소프트웨어", "개발자", "연구", "IT", "엔지니어"])
    is_young_heavy = youth_ratio > 0.40

    def w(*factors) -> float:
        return round(max(0.1, sum(factors)), 2)

    comfort_weights = {
        # Temperature — universal
        "too_hot":        1.0,
        "too_cold":       1.0,
        "stuffy":         0.8,
        "hot_drink_warm": w(0.4, youth_ratio * 0.8),
        "cold_draft":     w(0.4, senior_ratio * 1.2),
        # Lighting — tech/study districts feel it more
        "too_dark":       w(0.5, 1.5 if (is_tech or is_young_heavy) else 0.3),
        "too_bright":     w(0.4, senior_ratio * 1.8),
        # Noise — younger districts more noise-sensitive
        "noisy":          w(0.2, youth_ratio * 2.0),
        "music_low":      w(0.2, youth_ratio * 1.2),
        # Service — dirty table bothers older customers more
        "no_cups":        1.0,
        "dirty_table":    w(0.4, senior_ratio * 2.0),
        # Tech needs — strongly tied to youth + occupation
        "wifi_slow":      w(0.3, (2.0 if is_tech else 0.6), youth_ratio * 1.0),
        "outlet_needed":  w(0.2, (2.2 if is_tech else 0.4), youth_ratio * 1.2),
    }

    safety_weights = {
        # Thermal distress — elderly more vulnerable
        "sweat_wiping":       w(0.4, senior_ratio * 1.8),
        "heat_exhaustion":    w(0.2, senior_ratio * 3.5),
        # Theft — slightly higher where youth/students are prevalent
        "theft_attempt":      1.0,
        "kiosk_bypass":       w(0.4, youth_ratio * 1.2),
        "multiple_items":     1.0,
        # Property damage — younger demographics
        "property_damage":    w(0.2, youth_ratio * 2.0),
        "graffiti":           w(0.1, youth_ratio * 2.0),
        "equipment_tamper":   w(0.2, youth_ratio * 1.5),
        # Falls — core driver: elderly fall risk
        "fall_emergency":     w(0.3, senior_ratio * 5.0),
        "faint_suspected":    w(0.2, senior_ratio * 3.5),
        "slip_fall":          w(0.4, senior_ratio * 2.5),
        # Suspicious / harassment — universal
        "suspicious_person":  1.0,
        "loitering":          1.0,
        "aggressive_behavior":1.0,
        "harassment":         1.0,
        "verbal_harassment":  1.0,
    }

    # Overall probability: younger → more comfort complaints; older → more safety incidents
    comfort_prob = round(min(0.35, 0.12 + youth_ratio * 0.12), 3)
    safety_prob  = round(min(0.15, 0.04 + senior_ratio * 0.10), 3)

    return {
        "comfort_event_prob": comfort_prob,
        "safety_event_prob":  safety_prob,
        "comfort_weights":    comfort_weights,
        "safety_weights":     safety_weights,
        # Metadata for UI display
        "age_mean":           round(age_info.get("mean", 35.0), 1),
        "senior_ratio":       round(senior_ratio, 2),
        "youth_ratio":        round(youth_ratio, 2),
    }


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

async def generate_cafe_flow(profile: dict[str, Any]) -> dict[str, Any]:
    """상권 프로필 → 카페 흐름 JSON (LLM 생성)."""

    human_prompt = f"""
District profile:
{json.dumps(profile, ensure_ascii=False, indent=2)}

Generate the cafe flow configuration JSON for this district.
Remember:
- All text fields must be in Korean with emojis in action labels.
- Use exact field names: "id", "probability", "duration_range", "zone", "actions", "label"
- Do NOT use "stage" or "prob" as field names.
- Every stage must have an "actions" array with 3-5 Korean strings.
"""

    try:
        raw = await asyncio.to_thread(
            client.responses.create,
            model=os.getenv("CAFE_FLOW_MODEL", "gpt-4o-mini"),
            instructions=SYSTEM_PROMPT,
            input=human_prompt,
            text={"format": {"type": "json_object"}},
            service_tier="default",
            store=False,
        )
        content = raw.output_text.strip()
        print(f"[cafe_flow] ✅ LLM raw output:\n{content[:500]}")

    except Exception as e:
        print(f"[cafe_flow] ❌ LLM call failed: {e}")
        result = _patch_defaults({})
        result["event_config"] = _compute_event_config(profile)
        return result

    # Strip markdown fences if LLM wraps anyway
    content = re.sub(r"^```(?:json)?\s*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n?```\s*$",          "", content, flags=re.MULTILINE)
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[cafe_flow] ❌ JSON parse failed: {e}\n---\n{content}")
        return _patch_defaults({})

    print(f"[cafe_flow] ✅ Parsed {len(data.get('flow_stages', []))} stages, "
          f"{len(data.get('persona_types', []))} personas")

    result = _patch_defaults(data)
    result["event_config"] = _compute_event_config(profile)
    print(f"[cafe_flow] 📊 event_config: comfort_prob={result['event_config']['comfort_event_prob']}, "
          f"safety_prob={result['event_config']['safety_event_prob']}, "
          f"senior={result['event_config']['senior_ratio']}, youth={result['event_config']['youth_ratio']}")
    return result
