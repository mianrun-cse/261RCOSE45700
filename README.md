# 무인 스크린골프장 AI 관리 시스템

LangGraph 기반 멀티에이전트 아키텍처로 구현된 무인 스크린골프장 통합 관리 시스템.  
영상 안전 감지, 고객 응대, 스윙 코칭, 일일 보고서를 역할별 전문 에이전트가 처리하며 오케스트레이터가 에이전트 간 협력을 조율한다.

---

## 아키텍처

```
                    [관리자 (Human)]
                          ↑ SMS / 푸시 알림
               [오케스트레이터 에이전트] ── gpt-4o
              /       |               \
    [안전 에이전트]  [고객봇 에이전트]  [보고서 에이전트]
    gpt-4o-mini      gpt-4o-mini      gpt-4o-mini
    (→ gpt-4o 자율    (베이별 N개)
       에스컬레이션)
         ↕ P2P
    [코칭 에이전트]
    gpt-4o-mini

Tools (에이전트 아님):
  알림 발송 (SMS/푸시) │ TTS 음성 생성 │ 환경 제어 API │ SQLite 로깅
```

### 에이전트별 역할

| 에이전트 | 파일 | 모델 | 역할 |
|---------|------|------|------|
| 오케스트레이터 | `agents/orchestrator.py` | gpt-4o | 크로스베이 조율, 충돌 중재, 관리자 에스컬레이션 |
| 안전 감지 | `agents/safety_agent.py` | gpt-4o-mini → gpt-4o | 영상 분석, confidence 기반 모델 자율 에스컬레이션 |
| 고객봇 | `agents/customer_agent.py` | gpt-4o-mini | 자연어 요청 처리, 환경 제어, 크로스베이 요청 감지 |
| 코칭 | `agents/coaching_agent.py` | gpt-4o-mini | 스윙 자세 분석, 피드백 TTS 생성 |
| 보고서 | `agents/report_agent.py` | gpt-4o-mini | 일일 운영 데이터 요약, 이상 패턴 감지 |

### 라우팅 규칙

```
safety   → [오케스트레이터] : 고위험 + confidence < 0.80 (불확실)
         → [END]           : 정상 감지 (알림/온도제어 직접 실행)

customer → [오케스트레이터] : "전체/모든 타석" 키워드 감지 (크로스베이)
         → [END]           : 단일 베이 요청 (즉시 실행)

coaching → [END]           : 항상 직접 종료

report   → [오케스트레이터] : 이상 패턴 키워드 감지
         → [END]           : 정상 보고서
```

---

## 프로젝트 구조

```
실전SW/
├── main.py                        # 진입점 — 베이 루프 + 보고서 루프
├── requirements.txt
├── .env.example
│
├── llm_module/
│   ├── graph.py                   # LangGraph StateGraph 정의
│   ├── state.py                   # FacilityState 스키마 + 팩토리 함수
│   ├── state_machine.py           # VLM 빈도 제어 / cooldown (베이별)
│   ├── customer_bot.py            # 고객봇 public API (graph 경유)
│   ├── vlm_analyzer.py            # OpenAI Vision API 래퍼
│   ├── coaching_engine.py         # 자세 분석 + TTS
│   ├── report_generator.py        # 일일 리포트 생성
│   ├── alert_manager.py           # SMS / 푸시 알림 Tool
│   ├── temperature_controller.py  # 환경 제어 API Tool
│   └── agents/
│       ├── orchestrator.py        # 오케스트레이터 에이전트
│       ├── safety_agent.py        # 안전 감지 에이전트
│       ├── customer_agent.py      # 고객봇 에이전트
│       ├── coaching_agent.py      # 코칭 에이전트
│       └── report_agent.py        # 보고서 에이전트
│
├── db/
│   └── models.py                  # SQLite 스키마 + 쿼리
├── opencv/
│   ├── bridge.py                  # OpenCV ↔ asyncio 어댑터
│   └── face_detection.py          # MediaPipe 감지
└── test_modules.py                # 모듈별 테스트 스크립트
```

---

## 설치 및 실행

### 요구사항

- Python 3.11+
- OpenAI API 키
- (선택) Solapi SMS 키, 환경 제어 API

### 설치

```bash
pip install -r requirements.txt
```

### 환경변수 설정

```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

주요 환경변수:

| 변수 | 설명 | 기본값 |
|-----|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 |
| `VISION_MODEL` | 안전 감지 1차 모델 | `gpt-4o-mini` |
| `ESCALATION_MODEL` | 안전 감지 에스컬레이션 모델 | `gpt-4o` |
| `TTS_MODEL` | TTS 모델 | `tts-1` |
| `TTS_VOICE` | TTS 음성 | `nova` |
| `CLOSE_HOUR` | 보고서 생성 시각 (24h) | `22` |

### 실행

```bash
python main.py
```

---

## 테스트

```bash
python test_modules.py
```

```
테스트할 모듈을 선택하세요:
  [1] 고객 응대 챗봇          ← facility_graph 경유 (에이전트 로그 출력)
  [2] 자세 피드백 + TTS
  [3] 일일 리포트
  [4] 관리자 알림
  [5] 멀티에이전트 시나리오    ← 오케스트레이터 개입 여부 확인
  [6] 전체 실행
```

**[5] 멀티에이전트 시나리오** 테스트는 다음 3가지 경로를 검증한다:

| 시나리오 | 입력 | 기대 경로 |
|---------|------|---------|
| 단일 베이 요청 | "온도 낮춰줘" | customer → END |
| 전체 타석 요청 | "전체 타석 온도 22도로" | customer → orchestrator → END |
| 고위험 불확실 감지 | confidence=0.72, severity=high | safety → orchestrator → END |

---

## 주요 설계 결정

- **하이브리드 오케스트레이션** — 단순 P2P 통신은 오케스트레이터 우회, 크로스베이/충돌만 개입
- **모델 자율 에스컬레이션** — 안전 에이전트가 confidence 보고 스스로 gpt-4o로 재판단
- **Human-in-the-loop** — 에이전트 간 판단 충돌 시 관리자에게 알림, AI가 최종 결정하지 않음
- **Tool vs Agent 구분** — 판단/추론 필요 → 에이전트, 단순 실행(알림/TTS/환경제어) → Tool
- **기존 async 구조 유지** — LangGraph `ainvoke()`가 asyncio 루프에 자연스럽게 통합
