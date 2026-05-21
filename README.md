# 무인 매장 AI 관리 시스템

LangGraph 기반 멀티에이전트 아키텍처로 구현된 무인 매장(무인 카페·편의점 등) 통합 관리 시스템.
영상 안전 감지, 고객 응대, 일일 보고서를 역할별 전문 에이전트가 처리하며, 오케스트레이터가 에이전트 간 협력을 조율한다.
모든 에이전트는 **부수효과 없는 "결정 노드"**이며, 실제 실행(알림·환경 제어·TTS)은 그래프 말단의 `actuator` 노드가 전담한다.

---

## 아키텍처

```
                    [관리자 (Human)]
                          ↑ SMS / 푸시 알림
               [오케스트레이터 에이전트] ── gpt-4o
              /          |            \
    [안전 에이전트]  [고객봇 에이전트]  [보고서 에이전트]
    gpt-4o-mini      gpt-4o-mini      gpt-4o-mini
    (→ gpt-4o 자율    (구역별 N개)
       에스컬레이션)
              \          |            /
               [실행(actuator) 노드]   ← 모든 부수효과는 여기서만 실행
                          ↓
              알림 발송 │ 환경 제어 │ TTS 음성 │ 관리자 푸시

설계 원칙: 결정(에이전트) ↔ 실행(actuator) 분리
```

### 에이전트별 역할

| 에이전트 | 파일 | 모델 | 역할 |
|---------|------|------|------|
| 오케스트레이터 | `agents/orchestrator.py` | gpt-4o | 크로스존 요청 판단, 충돌 중재, 관리자 에스컬레이션 결정 |
| 안전 감지 | `agents/safety_agent.py` | gpt-4o-mini → gpt-4o | 영상 분석, confidence 기반 모델 자율 에스컬레이션 |
| 고객봇 | `agents/customer_agent.py` | gpt-4o-mini | 자연어 요청 처리, 환경 제어 판단, 크로스존 요청 감지 |
| 보고서 | `agents/report_agent.py` | gpt-4o-mini | 일일 운영 데이터 요약, 이상 패턴 감지 |
| 실행(actuator) | `agents/actuator.py` | — (LLM 미사용) | 결정 노드가 발행한 `pending_actions` 일괄 실행 |

> 결정 노드(안전/고객/보고서/오케스트레이터)는 알림·온도 제어·TTS 같은 부수효과를 직접 호출하지 않는다.
> 대신 실행 의도를 `state["pending_actions"]`에 적고, `actuator` 노드가 그래프 말단에서 이를 일괄 실행한다.

### 라우팅 규칙

```
safety   → [오케스트레이터] : 고위험 + confidence < 0.80 (불확실)
         → [actuator]      : 정상 감지 (알림/온도제어 의도 발행)

customer → [오케스트레이터] : "전체/모든 구역" 키워드 감지 (크로스존)
         → [actuator]      : 단일 구역 요청

report   → [오케스트레이터] : 이상 패턴 키워드 감지
         → [actuator]      : 정상 보고서

orchestrator → [actuator]  : 항상
actuator     → [END]       : 부수효과 실행 후 종료
```

---

## 프로젝트 구조

```
실전SW/
├── main.py                        # 진입점 — 구역 루프 + 보고서 루프
├── requirements.txt
│
├── llm_module/
│   ├── graph.py                   # LangGraph StateGraph 정의
│   ├── state.py                   # FacilityState 스키마 + 팩토리 함수
│   ├── state_machine.py           # VLM 빈도 제어 / cooldown (구역별)
│   ├── customer_bot.py            # 고객봇 public API (graph 경유)
│   ├── vlm_analyzer.py            # OpenAI Vision API 래퍼
│   ├── report_generator.py        # 일일 리포트 생성
│   ├── tts.py                     # 공용 TTS 유틸리티
│   ├── alert_manager.py           # SMS / 푸시 알림 Tool
│   ├── temperature_controller.py  # 환경 제어 API Tool
│   └── agents/
│       ├── orchestrator.py        # 오케스트레이터 에이전트 (결정)
│       ├── safety_agent.py        # 안전 감지 에이전트 (결정)
│       ├── customer_agent.py      # 고객봇 에이전트 (결정)
│       ├── report_agent.py        # 보고서 에이전트 (결정)
│       └── actuator.py            # 실행 노드 (부수효과 전담)
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

프로젝트 루트에 `.env` 파일을 만들고 아래 키를 입력한다:

```bash
OPENAI_API_KEY=sk-...
```

주요 환경변수:

| 변수 | 설명 | 기본값 |
|-----|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 |
| `VISION_MODEL` | 안전 감지 1차 모델 | `gpt-4o-mini` |
| `ESCALATION_MODEL` | 안전 감지 에스컬레이션 모델 | `gpt-4o` |
| `TTS_MODEL` | TTS 모델 | `tts-1` |
| `TTS_VOICE` | TTS 음성 | `nova` |
| `DB_PATH` | SQLite 파일 경로 | `store.db` |
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
  [2] 일일 리포트
  [3] 관리자 알림
  [4] 멀티에이전트 시나리오    ← 오케스트레이터 / actuator 개입 여부 확인
  [5] 전체 실행
```

**[4] 멀티에이전트 시나리오** 테스트는 다음 3가지 경로를 검증한다:

| 시나리오 | 입력 | 기대 경로 |
|---------|------|---------|
| 단일 구역 요청 | "온도 낮춰줘" | customer → actuator → END |
| 전체 구역 요청 | "전체 구역 온도 22도로" | customer → orchestrator → actuator → END |
| 고위험 불확실 감지 | confidence=0.72, severity=high | safety → orchestrator → actuator → END |

---

## 주요 설계 결정

- **결정/실행 분리** — 에이전트 노드는 판단만 하고 `pending_actions`로 의도를 발행, `actuator` 노드가 부수효과를 전담. 결정 로직을 외부 API 호출 없이 단위 테스트 가능
- **하이브리드 오케스트레이션** — 단순 P2P 통신은 오케스트레이터 우회, 크로스존/충돌만 개입
- **모델 자율 에스컬레이션** — 안전 에이전트가 confidence 보고 스스로 gpt-4o로 재판단
- **Human-in-the-loop** — 에이전트 간 판단 충돌 시 관리자에게 알림, AI가 최종 결정하지 않음
- **누적 액션 리듀서** — `pending_actions`는 `operator.add` 리듀서로 여러 노드(customer→orchestrator)의 발행이 합쳐짐
- **기존 async 구조 유지** — LangGraph `ainvoke()`가 asyncio 루프에 자연스럽게 통합
