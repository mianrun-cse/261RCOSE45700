"""
face_detection.py의 occlusion_active 신호와 BodySwayDetector를
state_machine 큐에 연결하는 어댑터.

USB 웹캠(int) 또는 Tapo RTSP URL(str)을 지원한다.
"""
import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import asyncio
import base64
import threading
import time
from collections import deque
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from openai import OpenAI

from llm_module.state_machine import TriggerSignals

OPENCV_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(OPENCV_DIR, 'hand_landmarker.task')
POSE_MODEL_PATH  = os.path.join(OPENCV_DIR, 'pose_landmarker_lite.task')
FACE_MODEL_PATH  = os.path.join(OPENCV_DIR, 'blaze_face_short_range.tflite')
FACE_LANDMARKER_PATH = os.path.join(OPENCV_DIR, 'face_landmarker.task')
DANGER_SCREENSHOT_DIR = 'danger_screenshots'

# 한글 텍스트 렌더링 — cv2.putText는 ASCII Hershey 폰트만 지원해 한글이 깨지므로 PIL로 그린다.
KOREAN_FONT_PATH = os.getenv("KOREAN_FONT", "C:/Windows/Fonts/malgun.ttf")
_font_cache: dict[int, "ImageFont.FreeTypeFont"] = {}


def _get_font(size: int) -> "ImageFont.FreeTypeFont":
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(KOREAN_FONT_PATH, size)
        except OSError:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_text_kr(img, text, org, size=22, color=(0, 255, 0)):
    """cv2.putText의 한글 지원 대체. img를 in-place 수정한다. color는 BGR(OpenCV 관례)."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(org, text, font=_get_font(size), fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def put_multiline_kr(img, text, org, size=20, color=(255, 255, 255), max_chars=34, line_gap=6):
    """긴 한글 문장을 글자 수 기준으로 줄바꿈하여 그린다. (PIL 변환 1회로 처리)"""
    x, y = org
    font = _get_font(size)
    lines = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [""]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fill = (color[2], color[1], color[0])
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += size + line_gap
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

FRAME_SIZE                 = (640, 360)
OCCLUSION_TRIGGER_FRAMES   = 10
PERSON_GONE_TRIGGER_FRAMES = 30
SAMPLE_INTERVAL_SEC        = 0.4

# 몸 흔들림 감지
SWAY_WINDOW_FRAMES      = 15
SWAY_X_THRESHOLD        = 30
SWAY_Y_THRESHOLD        = 25
SWAY_TRIGGER_FRAMES     = 5
SWAY_CAPTURE_COUNT      = 10
SWAY_CAPTURE_INTERVAL   = 3
OPENAI_API_COOLDOWN_SEC = 30
_POSE_LEFT_SHOULDER     = 11
_POSE_RIGHT_SHOULDER    = 12
_HAND_WRIST             = 0

# 손 흔들림 감지
HAND_SHAKE_WINDOW_FRAMES  = 8
HAND_SHAKE_THRESHOLD      = 40   # px — 손 중심이 이 이상 흔들리면 감지
HAND_SHAKE_TRIGGER_FRAMES = 3

# 입 모양(립 무브먼트) 감지 → 대화 트리거
#   MAR(Mouth Aspect Ratio) = 입 세로열림 / 가로폭. 말할 때 MAR이 진동하므로
#   윈도우 내 표준편차가 임계값을 넘으면 "말하는 중"으로 판정한다.
MOUTH_WINDOW_FRAMES       = 12
MOUTH_OPEN_STD_THRESHOLD  = 0.030  # MAR 표준편차 임계 — 낮추면 민감, 높이면 둔감
MOUTH_TALK_TRIGGER_FRAMES = 5      # 연속 N프레임 진동하면 대화 시작

# 음성 대화 루프
CONVERSATION_RECORD_SEC   = 5.0    # 한 턴당 녹음 길이
CONVERSATION_MAX_TURNS    = 6      # 한 세션 최대 대화 턴
CONVERSATION_COOLDOWN_SEC = 20.0   # 대화 종료 후 재트리거 금지 시간

os.makedirs(DANGER_SCREENSHOT_DIR, exist_ok=True)

CameraSource = Union[int, str]


# ═══════════════════════════════════════════════════════════════
#  몸 흔들림 감지
# ═══════════════════════════════════════════════════════════════

class BodySwayDetector:
    """어깨 중심점을 추적해 과도한 몸 흔들림을 감지한다."""

    def __init__(self):
        self.left_shoulder_history: list[tuple]  = []
        self.right_shoulder_history: list[tuple] = []
        self.sway_consecutive   = 0
        self.capturing          = False
        self.capture_frames: list = []
        self.capture_countdown  = 0
        self.last_api_call_time = 0.0
        self.hand_position_history: list[tuple] = []
        self.hand_shake_consecutive = 0

    def _shoulder_positions(self, pose_results, frame_shape) -> tuple | None:
        """왼쪽·오른쪽 어깨 개별 픽셀 좌표 ((lx,ly),(rx,ry)) 반환."""
        if not pose_results.pose_landmarks:
            return None
        lm    = pose_results.pose_landmarks[0]
        h, w  = frame_shape[:2]
        left  = lm[_POSE_LEFT_SHOULDER]
        right = lm[_POSE_RIGHT_SHOULDER]
        if left.visibility < 0.5 or right.visibility < 0.5:
            return None
        return (int(left.x * w), int(left.y * h)), (int(right.x * w), int(right.y * h))

    def _shoulder_range(self, history: list[tuple]) -> tuple[int, int]:
        if len(history) < 2:
            return 0, 0
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        return max(xs) - min(xs), max(ys) - min(ys)

    def _hand_center(self, hand_results, frame_shape) -> tuple | None:
        """감지된 손목들의 평균 위치 반환."""
        if not hand_results or not hand_results.hand_landmarks:
            return None
        h, w = frame_shape[:2]
        xs, ys = [], []
        for hand_lms in hand_results.hand_landmarks:
            wrist = hand_lms[_HAND_WRIST]
            xs.append(int(wrist.x * w))
            ys.append(int(wrist.y * h))
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def _is_hand_shaking(self) -> bool:
        if len(self.hand_position_history) < HAND_SHAKE_WINDOW_FRAMES:
            return False
        xs = [p[0] for p in self.hand_position_history]
        ys = [p[1] for p in self.hand_position_history]
        return (max(xs) - min(xs) > HAND_SHAKE_THRESHOLD or
                max(ys) - min(ys) > HAND_SHAKE_THRESHOLD)

    def get_hand_shake_range(self) -> tuple[int, int]:
        if len(self.hand_position_history) < 2:
            return 0, 0
        xs = [p[0] for p in self.hand_position_history]
        ys = [p[1] for p in self.hand_position_history]
        return max(xs) - min(xs), max(ys) - min(ys)

    def _is_swaying(self) -> bool:
        if (len(self.left_shoulder_history) < SWAY_WINDOW_FRAMES or
                len(self.right_shoulder_history) < SWAY_WINDOW_FRAMES):
            return False
        lx, ly = self._shoulder_range(self.left_shoulder_history)
        rx, ry = self._shoulder_range(self.right_shoulder_history)
        left_sway  = lx > SWAY_X_THRESHOLD or ly > SWAY_Y_THRESHOLD
        right_sway = rx > SWAY_X_THRESHOLD or ry > SWAY_Y_THRESHOLD
        return left_sway and right_sway

    def get_sway_range(self) -> tuple[int, int, int, int]:
        """(왼X범위, 왼Y범위, 오른X범위, 오른Y범위)"""
        lx, ly = self._shoulder_range(self.left_shoulder_history)
        rx, ry = self._shoulder_range(self.right_shoulder_history)
        return lx, ly, rx, ry

    def update(self, pose_results, frame, frame_shape, hand_results=None) -> str | None:
        """
        Returns: 'trigger' | 'capturing' | 'ready' | None
        """
        positions = self._shoulder_positions(pose_results, frame_shape)
        if positions is not None:
            left_pos, right_pos = positions
            self.left_shoulder_history.append(left_pos)
            self.right_shoulder_history.append(right_pos)
            if len(self.left_shoulder_history) > SWAY_WINDOW_FRAMES:
                self.left_shoulder_history.pop(0)
            if len(self.right_shoulder_history) > SWAY_WINDOW_FRAMES:
                self.right_shoulder_history.pop(0)
        else:
            self.left_shoulder_history.clear()
            self.right_shoulder_history.clear()
            self.sway_consecutive = 0

        # 손 위치 이력 업데이트
        hand_center = self._hand_center(hand_results, frame_shape)
        if hand_center is not None:
            self.hand_position_history.append(hand_center)
            if len(self.hand_position_history) > HAND_SHAKE_WINDOW_FRAMES:
                self.hand_position_history.pop(0)
        else:
            self.hand_position_history.clear()
            self.hand_shake_consecutive = 0

        if self.capturing:
            self.capture_countdown -= 1
            if self.capture_countdown <= 0:
                self.capture_frames.append(frame.copy())
                self.capture_countdown = SWAY_CAPTURE_INTERVAL
                if len(self.capture_frames) >= SWAY_CAPTURE_COUNT:
                    self.capturing = False
                    self.sway_consecutive = 0
                    return 'ready'
            return 'capturing'

        if self._is_swaying():
            self.sway_consecutive += 1
        else:
            self.sway_consecutive = max(0, self.sway_consecutive - 1)

        if self._is_hand_shaking():
            self.hand_shake_consecutive += 1
        else:
            self.hand_shake_consecutive = max(0, self.hand_shake_consecutive - 1)

        now = time.time()
        hand_shaking = self.hand_shake_consecutive >= HAND_SHAKE_TRIGGER_FRAMES
        if (self.sway_consecutive >= SWAY_TRIGGER_FRAMES
                and hand_shaking
                and now - self.last_api_call_time > OPENAI_API_COOLDOWN_SEC):
            self.capturing = True
            self.capture_frames = [frame.copy()]
            self.capture_countdown = SWAY_CAPTURE_INTERVAL
            print(f"[DANGER] 몸+손 흔들림 동시 감지! 캡처 시작 "
                  f"(어깨={self.sway_consecutive}f, 손={self.hand_shake_consecutive}f)")
            return 'trigger'

        return None

    def get_captured_frames(self) -> list:
        return list(self.capture_frames)

    def reset_capture(self):
        self.capture_frames.clear()
        self.last_api_call_time = time.time()


# ═══════════════════════════════════════════════════════════════
#  얼굴 모자이크 + 위험 스크린샷 저장
# ═══════════════════════════════════════════════════════════════

def apply_face_mosaic(frame, face_detector, mosaic_scale=0.05):
    """프레임의 모든 얼굴 영역에 픽셀 모자이크를 적용한 복사본 반환."""
    if face_detector is None:
        return frame.copy()
    fh_img, fw_img = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res    = face_detector.detect(mp_img)
    result = frame.copy()
    if res.detections:
        for det in res.detections:
            bb = det.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(fw_img, bb.origin_x + bb.width)
            y2 = min(fh_img, bb.origin_y + bb.height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi     = result[y1:y2, x1:x2]
            rh, rw  = roi.shape[:2]
            small_w = max(1, int(rw * mosaic_scale))
            small_h = max(1, int(rh * mosaic_scale))
            small   = cv2.resize(roi, (small_w, small_h))
            mosaic  = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            result[y1:y2, x1:x2] = mosaic
    return result


def save_danger_screenshots(frames, face_detector) -> list[str]:
    """위험 감지 프레임들을 얼굴 모자이크 처리 후 JPEG로 저장. 경로 목록 반환."""
    from datetime import datetime
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = []
    for i, frame in enumerate(frames):
        mosaiced = apply_face_mosaic(frame, face_detector)
        path     = os.path.join(DANGER_SCREENSHOT_DIR, f'danger_{timestamp}_{i + 1}.jpg')
        cv2.imwrite(path, mosaiced)
        saved_paths.append(path)
        print(f"[DANGER] 저장: {path}")
    return saved_paths


def analyze_danger_with_ai_api(image_paths: list[str], result_callback):
    """모자이크 처리된 이미지를 GPT-4o로 분석. 백그라운드 스레드에서 실행."""
    # 파일 읽기 + 즉시 삭제 (API 성공/실패와 무관하게 삭제 보장)
    content = []
    for path in image_paths:
        img = cv2.imread(path)
        try:
            os.remove(path)
        except OSError:
            pass
        if img is None:
            continue
        _, buf  = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    if not content:
        result_callback("Failed to load images.")
        return

    try:
        client = OpenAI()

        content.append({
            "type": "text",
            "text": (
                "The following images are consecutive frames from an indoor CCTV monitoring system. "
                "An anomaly (excessive body sway or sudden movement) was automatically detected. "
                "Faces have been intentionally blurred to protect personal privacy. "
                "Face identification is NOT required and NOT requested — analyze only body posture, "
                "movement, physical interaction between people, and the overall situation.\n\n"
                "답변의 맨 첫 줄에는 반드시 다음 형식으로만 출력하세요: "
                "'위험판정 결과 : 위험함 (신뢰도 0.0~1.0)' 또는 "
                "'위험판정 결과 : 위험하지 않음 (신뢰도 0.0~1.0)'. "
                "신뢰도는 판정에 대한 확신 정도이며, 영상이 흐릿하거나 정보가 부족하면 "
                "낮은 신뢰도(예: 0.3)로, 명확하면 높은 신뢰도로 적으세요. "
                "그 다음 줄부터 아래 항목에 답하세요.\n"
                "다음 항목에 간결하게 한국어로 답해 주세요. 영상이 흐릿하거나 불확실하더라도 "
                "거절하지 말고, 단정하기 어려우면 낮은 신뢰도로 '가장 가능성 높은 해석'을 제시하세요. "
                "'판단 불가'라고만 답하고 끝내지는 마세요.\n"
                "1. 위험한 상황인가요? (예 / 아니오)\n"
                "2. 폭력적인 상황인가요? (예 / 아니오) — 폭행, 다툼, 밀침, 가격, 공격 등의 정황이 보이면 그 근거를 적으세요.\n"
                "3. 관찰된 위험 행동이나 신체 상태는 무엇인가요? "
                "(예: 비틀거림, 쓰러짐, 발작, 몸싸움, 가격, 제압 등)\n"
                "4. 즉각적인 도움이 필요한가요?\n"
                "5. 어떤 상황인지 설명하세요.\n"
                "답변 할 때는 질문도 포함해서 대답하세요."
            ),
        })

        response = client.chat.completions.create(
            model="gpt-5",
            max_completion_tokens=2048,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an authorized safety-monitoring assistant for an indoor CCTV "
                        "emergency-response system. Your only purpose is to protect the safety of "
                        "residents by assessing whether a fall, medical emergency, or physical "
                        "violence is occurring. All faces are blurred and you must never attempt to "
                        "identify individuals. You must always provide a best-effort safety "
                        "assessment for every set of frames, even when the images are ambiguous, "
                        "low quality, or distressing. Do not refuse; instead state your most "
                        "likely interpretation together with an explicit confidence score (0.0-1.0). "
                        "When the frames are ambiguous, give a low-confidence best guess rather than "
                        "a confident verdict — never assert danger or safety you cannot support. "
                        "Answer in Korean."
                    ),
                },
                {"role": "user", "content": content},
            ],
        )
        result_callback(response.choices[0].message.content)

    except Exception as exc:
        result_callback(f"[오류] OpenAI API 호출 실패: {exc}")


# ═══════════════════════════════════════════════════════════════
#  카메라 유틸
# ═══════════════════════════════════════════════════════════════

def _open_capture(camera_source: CameraSource) -> cv2.VideoCapture:
    if isinstance(camera_source, str):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(camera_source)
    return cap


def _camera_label(camera_source: CameraSource) -> str:
    if isinstance(camera_source, str):
        return "RTSP stream"
    return f"USB camera ({camera_source})"


def _hand_near_face(hand_results, face_region, frame_shape, margin_ratio=0.6) -> bool:
    """얼굴 영역 근처에 손 랜드마크가 있는지 확인."""
    if not hand_results.hand_landmarks or face_region is None:
        return False
    h_img, w_img = frame_shape[:2]
    fx, fy, fw, fh = face_region
    margin = int(max(fw, fh) * margin_ratio)
    rx1, ry1 = max(0, fx - margin), max(0, fy - margin)
    rx2, ry2 = min(w_img, fx + fw + margin), min(h_img, fy + fh + margin)
    for hand_lms in hand_results.hand_landmarks:
        for lm in hand_lms:
            hx, hy = int(lm.x * w_img), int(lm.y * h_img)
            if rx1 <= hx <= rx2 and ry1 <= hy <= ry2:
                return True
    return False


# ═══════════════════════════════════════════════════════════════
#  메인 감지 루프
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  입 모양(립 무브먼트) 감지 — 대화 시작 트리거
# ═══════════════════════════════════════════════════════════════
# MediaPipe FaceLandmarker(478 landmark)의 입술 랜드마크 기준
_LIP_TOP    = 13    # 윗입술 안쪽 중앙
_LIP_BOTTOM = 14    # 아랫입술 안쪽 중앙
_LIP_LEFT   = 78    # 입꼬리 좌
_LIP_RIGHT  = 308   # 입꼬리 우


class MouthMovementDetector:
    """
    얼굴 랜드마크에서 입 벌림 비율(MAR)을 추적해 '말하는 중'인지 판정한다.
    말할 때는 MAR이 빠르게 진동(개폐)하므로, 최근 윈도우의 표준편차가
    임계값을 넘는 프레임이 연속으로 누적되면 talking=True 를 반환한다.
    """

    def __init__(self):
        self.mar_window      = deque(maxlen=MOUTH_WINDOW_FRAMES)
        self.talk_consecutive = 0
        self.last_mar        = 0.0
        self.last_std        = 0.0

    def update(self, face_landmarker_result) -> bool:
        landmarks = getattr(face_landmarker_result, "face_landmarks", None)
        if not landmarks:
            self.mar_window.clear()
            self.talk_consecutive = 0
            self.last_std = 0.0
            return False

        lm     = landmarks[0]
        vert   = abs(lm[_LIP_TOP].y - lm[_LIP_BOTTOM].y)
        horiz  = abs(lm[_LIP_LEFT].x - lm[_LIP_RIGHT].x) or 1e-6
        mar    = vert / horiz
        self.last_mar = mar
        self.mar_window.append(mar)

        if len(self.mar_window) < self.mar_window.maxlen:
            return False

        std = float(np.std(self.mar_window))
        self.last_std = std
        if std >= MOUTH_OPEN_STD_THRESHOLD:
            self.talk_consecutive += 1
        else:
            self.talk_consecutive = max(0, self.talk_consecutive - 1)

        return self.talk_consecutive >= MOUTH_TALK_TRIGGER_FRAMES


def _danger_verdict_line(text: str) -> str:
    """AI 응답에서 '위험판정 결과 : ...' 한 줄만 추출. 없으면 본문에서 추정."""
    if text:
        for line in text.splitlines():
            if "위험판정 결과" in line:
                return line.strip()
        compact = text.replace(" ", "")
        if "위험하지않" in compact:
            return "위험판정 결과 : 위험하지 않음"
        if "위험함" in compact or "위험합니다" in compact:
            return "위험판정 결과 : 위험함"
    return "위험판정 결과 : 판정 불가"


async def _run_conversation(zone_id: str, conv_state: dict) -> None:
    """
    입 모양 감지로 시작되는 음성 대화 루프 (메인 asyncio 루프에서 실행).
    인사 → STT(Whisper) → customer_bot.respond → TTS 재생을, 무음이 나오거나
    최대 턴에 도달할 때까지 반복한다. 종료 시 conv_state 를 정리한다.
    """
    try:
        from llm_module.stt import record_and_transcribe, play_audio
        from llm_module.customer_bot import respond
    except ImportError as e:
        print(f"[{zone_id}][CONV] 음성 모듈 import 실패: {e} → pip install sounddevice soundfile")
        conv_state['active'] = False
        conv_state['last_end'] = time.time()
        return

    context = {
        "customer_name": "고객",
        "visit_count": 1,
        "current_temp": 25.0,
        "remaining_min": 30,
        "reserved_min": 60,
    }

    try:
        # 인사말도 하드코딩하지 않고 OpenAI가 상황을 받아 직접 생성 (프롬프트로 위임)
        try:
            greet = await respond(
                "고객이 다가와 말을 걸려고 합니다. 먼저 밝고 친절하게 인사하고, "
                "무엇을 도와드릴지 자연스럽게 한 문장으로 물어보세요.",
                zone_id, context, tts=True, all_zone_ids=[zone_id],
            )
            print(f"[{zone_id}][CONV] 봇(인사): {greet.get('message', '')}")
            if greet.get("audio_path"):
                await asyncio.to_thread(play_audio, greet["audio_path"])
        except Exception as e:
            print(f"[{zone_id}][CONV] 인사 생성 실패(무시): {e}")

        for _ in range(CONVERSATION_MAX_TURNS):
            text = await record_and_transcribe(seconds=CONVERSATION_RECORD_SEC)
            if not text:
                print(f"[{zone_id}][CONV] 무음 — 대화 종료")
                break
            print(f"[{zone_id}][CONV] 고객: {text}")

            result = await respond(text, zone_id, context, tts=True, all_zone_ids=[zone_id])
            msg = result.get("message", "")
            print(f"[{zone_id}][CONV] 봇: {msg}")

            audio_path = result.get("audio_path")
            if audio_path:
                await asyncio.to_thread(play_audio, audio_path)
    except Exception as e:
        print(f"[{zone_id}][CONV] 대화 오류: {e}")
    finally:
        conv_state['active']   = False
        conv_state['last_end'] = time.time()
        print(f"[{zone_id}][CONV] 세션 종료 (쿨다운 {CONVERSATION_COOLDOWN_SEC:.0f}초)")


def _run_detection_loop(
    zone_id: str,
    frame_queue: asyncio.Queue,
    signal_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    camera_source: CameraSource = 0,
) -> None:
    """동기 OpenCV 루프. 별도 스레드에서 실행된다."""
    if not os.path.exists(MODEL_PATH):
        print(f"[{zone_id}][BRIDGE] MediaPipe model not found: {MODEL_PATH}")
        print(f"[{zone_id}][BRIDGE] Download hand_landmarker.task to the opencv/ directory.")
        return

    # HandLandmarker
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

    # PoseLandmarker (VIDEO 모드 — 흔들림 감지)
    pose_detector = None
    if os.path.exists(POSE_MODEL_PATH):
        pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
    else:
        print(f"[{zone_id}][BRIDGE] {POSE_MODEL_PATH} 없음 → 몸 흔들림 감지 비활성화")

    # FaceDetector VIDEO 모드 — 실시간 얼굴 감지 (face_detection.py와 동일 엔진)
    face_detector_video = None
    # FaceDetector IMAGE 모드 — 모자이크 저장 전용
    face_detector_image = None
    if os.path.exists(FACE_MODEL_PATH):
        face_detector_video = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
        )
        face_detector_image = mp_vision.FaceDetector.create_from_options(
            mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
        )
    else:
        print(f"[{zone_id}][BRIDGE] {FACE_MODEL_PATH} 없음 → Haar Cascade 폴백, 모자이크 비활성화")

    # FaceLandmarker VIDEO 모드 — 입 모양(립 무브먼트) 감지로 대화 트리거
    face_landmarker_video = None
    if os.path.exists(FACE_LANDMARKER_PATH):
        face_landmarker_video = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
    else:
        print(f"[{zone_id}][BRIDGE] {FACE_LANDMARKER_PATH} 없음 → 입 모양 대화 트리거 비활성화")
        print(f"[{zone_id}][BRIDGE]   다운로드: https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = _open_capture(camera_source)
    if not cap.isOpened():
        print(f"[{zone_id}][BRIDGE] Failed to open {_camera_label(camera_source)}")
        if isinstance(camera_source, str):
            print(f"[{zone_id}][BRIDGE] Check TAPO_RTSP_URL in .env (username, password, IP, stream2).")
        return

    print(f"[{zone_id}][BRIDGE] Camera started ({_camera_label(camera_source)})")
    print(f"[{zone_id}][BRIDGE] ── 감지 설정 ──────────────────────────────────")
    print(f"[{zone_id}][BRIDGE]  손/얼굴 가림  : {'활성' if face_detector_video else 'Haar Cascade 폴백'}")
    print(f"[{zone_id}][BRIDGE]  몸+손 흔들림  : {'활성' if pose_detector else '비활성 (모델 없음)'}")
    print(f"[{zone_id}][BRIDGE]  얼굴 모자이크 : {'활성 — 위험 캡처 전 자동 처리' if face_detector_image else '비활성 (모델 없음)'}")
    print(f"[{zone_id}][BRIDGE]  위험 감지 시  : 모자이크 처리 후 OpenAI API 분석 요청 가능")
    print(f"[{zone_id}][BRIDGE]  API 쿨다운    : {OPENAI_API_COOLDOWN_SEC}초")
    print(f"[{zone_id}][BRIDGE] ──────────────────────────────────────────────")

    last_face_region      = None
    occlusion_frame_count = 0
    occlusion_active      = False
    person_gone_count     = 0

    frame_buffer       = []
    last_sample_time   = 0.0
    last_data_log_time = 0.0
    _loop_start_ms     = int(time.time() * 1000)

    sway_detector  = BodySwayDetector()
    danger_state   = {'pending': False, 'last_result': '', 'result_until': 0.0}

    mouth_detector = MouthMovementDetector()
    conv_state     = {'active': False, 'last_end': 0.0}

    def _on_danger_complete(result: str):
        verdict = _danger_verdict_line(result)
        print(f"\n{'=' * 52}")
        print(f"  [{zone_id}] {verdict}")
        print('-' * 52)
        print(result)
        print('=' * 52 + '\n')
        danger_state['pending'] = False
        # 화면 표시용: 간단한 판정 결과만 저장하고 10초간 노출
        danger_state['last_result'] = verdict
        danger_state['result_until'] = time.time() + 10.0

    debug      = os.getenv("DEBUG_CAMERA", "1") == "1"
    trace_data = os.getenv("TRACE_CAMERA_DATA", "1") == "1"
    use_rtsp   = isinstance(camera_source, str)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{zone_id}][BRIDGE] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            if use_rtsp or frame.shape[1] != FRAME_SIZE[0] or frame.shape[0] != FRAME_SIZE[1]:
                frame = cv2.resize(frame, FRAME_SIZE)

            now     = time.time()
            display = frame.copy()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(time.time() * 1000) - _loop_start_ms

            if face_detector_video is not None:
                fd_res = face_detector_video.detect_for_video(mp_image, timestamp_ms)
                faces  = [(d.bounding_box.origin_x, d.bounding_box.origin_y,
                           d.bounding_box.width, d.bounding_box.height)
                          for d in fd_res.detections] if fd_res.detections else []
            else:
                faces = list(face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)))

            hand_results = hands_detector.detect_for_video(mp_image, timestamp_ms)

            # Pose
            pose_results = None
            if pose_detector is not None:
                pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)

            # 얼굴/손 가림 감지
            # bridge.py는 LBPH 인식기가 없으므로 "얼굴이 감지된 상태에서 손이 근처에 있으면" occlusion으로 판정
            if len(faces) > 0:
                fx, fy, fw, fh   = max(faces, key=lambda f: f[2] * f[3])
                last_face_region = (fx, fy, fw, fh)
                person_gone_count = 0

                hand_detected = _hand_near_face(hand_results, last_face_region, frame.shape)
                if hand_detected:
                    occlusion_frame_count += 1
                    if occlusion_frame_count >= OCCLUSION_TRIGGER_FRAMES and not occlusion_active:
                        occlusion_active = True
                else:
                    occlusion_frame_count = max(0, occlusion_frame_count - 1)
                    if occlusion_active and occlusion_frame_count == 0:
                        occlusion_active = False

            elif last_face_region is not None:
                # 얼굴이 사라진 경우 — 일정 시간 후 상태 초기화
                person_gone_count += 1
                if person_gone_count >= PERSON_GONE_TRIGGER_FRAMES:
                    occlusion_active      = False
                    last_face_region      = None
                    occlusion_frame_count = 0
                    person_gone_count     = 0

            # 몸 흔들림 감지
            body_sway_signal = False
            if pose_results is not None:
                try:
                    sway_status = sway_detector.update(pose_results, frame, frame.shape, hand_results)

                    if sway_status == 'ready' and not danger_state['pending']:
                        captured    = sway_detector.get_captured_frames()
                        sway_detector.reset_capture()
                        saved_paths = save_danger_screenshots(captured, face_detector_image)
                        if saved_paths:
                            danger_state['pending'] = True
                            threading.Thread(
                                target=analyze_danger_with_ai_api,
                                args=(saved_paths, _on_danger_complete),
                                daemon=True,
                            ).start()
                            print(f"[{zone_id}][DANGER] OpenAI API 분석 요청 중...")

                    body_sway_signal = (
                        sway_detector.sway_consecutive >= SWAY_TRIGGER_FRAMES
                        and sway_detector.hand_shake_consecutive >= HAND_SHAKE_TRIGGER_FRAMES
                    )
                except Exception as e:
                    print(f"[{zone_id}][DANGER] 처리 오류 (카메라 유지): {e}")

            # 입 모양(립 무브먼트) 감지 → 대화 자동 시작
            talking = False
            if face_landmarker_video is not None:
                try:
                    fl_res  = face_landmarker_video.detect_for_video(mp_image, timestamp_ms)
                    talking = mouth_detector.update(fl_res)
                except Exception as e:
                    print(f"[{zone_id}][CONV] 입 모양 감지 오류 (카메라 유지): {e}")

            if (talking and not conv_state['active']
                    and now - conv_state['last_end'] >= CONVERSATION_COOLDOWN_SEC):
                conv_state['active'] = True
                print(f"[{zone_id}][CONV] 입 모양 감지 — 대화 시작")
                asyncio.run_coroutine_threadsafe(
                    _run_conversation(zone_id, conv_state), loop
                )

            # debug 시각화
            if debug:
                h_img, w_img = frame.shape[:2]
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if hand_results.hand_landmarks:
                    for hand_lms in hand_results.hand_landmarks:
                        for lm in hand_lms:
                            cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                            cv2.circle(display, (cx, cy), 4, (0, 215, 255), -1)
                if pose_results is not None and pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks[0]
                    ls = lm[_POSE_LEFT_SHOULDER]
                    rs = lm[_POSE_RIGHT_SHOULDER]
                    if ls.visibility >= 0.5:
                        cv2.circle(display, (int(ls.x * w_img), int(ls.y * h_img)), 8, (255, 100, 0), -1)
                    if rs.visibility >= 0.5:
                        cv2.circle(display, (int(rs.x * w_img), int(rs.y * h_img)), 8, (0, 140, 255), -1)

                alert = occlusion_active or body_sway_signal
                status = ("SWEAT!" if occlusion_active else "") + (" SWAY!" if body_sway_signal else "") or "monitoring..."
                cv2.putText(display, status.strip(), (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if alert else (0, 255, 0), 2)
                cv2.putText(display, f"occlusion: {occlusion_frame_count}/{OCCLUSION_TRIGGER_FRAMES}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"faces: {len(faces)}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if pose_results is not None:
                    lx, ly, rx, ry = sway_detector.get_sway_range()
                    cv2.putText(display,
                                f"L-Sway X:{lx} Y:{ly}  R-Sway X:{rx} Y:{ry} [{sway_detector.sway_consecutive}f]",
                                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
                    hx_range, hy_range = sway_detector.get_hand_shake_range()
                    cv2.putText(display,
                                f"Hand X:{hx_range} Y:{hy_range} [{sway_detector.hand_shake_consecutive}f]",
                                (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 255), 1)
                if sway_detector.capturing:
                    n = len(sway_detector.capture_frames)
                    cv2.putText(display, f"! CAPTURING {n}/{SWAY_CAPTURE_COUNT}",
                                (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if face_landmarker_video is not None:
                    cv2.putText(
                        display,
                        f"mouth std:{mouth_detector.last_std:.3f} "
                        f"[{mouth_detector.talk_consecutive}f] {'TALKING' if talking else ''}",
                        (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 1)
                if conv_state['active']:
                    cv2.putText(display, "CONVERSATION ACTIVE", (10, 235),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # AI 위험분석 결과 (한글) — 완료 후 일정 시간 화면 하단에 표시
                if danger_state['last_result'] and now < danger_state['result_until']:
                    put_multiline_kr(
                        display, f"[AI] {danger_state['last_result']}",
                        (10, h_img - 90), size=18, color=(0, 255, 255), max_chars=46,
                    )

                cv2.imshow(f"Bridge - {zone_id}", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 프레임 버퍼 (VLM용)
            if now - last_sample_time >= SAMPLE_INTERVAL_SEC:
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > 4:
                    frame_buffer.pop(0)
                last_sample_time = now

            signals = TriggerSignals(
                sweat_wiping=occlusion_active,
                body_sway=body_sway_signal,
                person_count=len(faces),
            )

            if trace_data and now - last_data_log_time >= 1.0:
                hand_count = len(hand_results.hand_landmarks or [])
                print(
                    f"[{zone_id}][DATA][BRIDGE->STATE] "
                    f"faces={len(faces)} hands={hand_count} "
                    f"occlusion_count={occlusion_frame_count}/{OCCLUSION_TRIGGER_FRAMES} "
                    f"sweat_wiping={signals.sweat_wiping} "
                    f"body_sway={signals.body_sway} "
                    f"person_count={signals.person_count} "
                    f"frame_buffer={len(frame_buffer)}/4"
                )
                last_data_log_time = now

            asyncio.run_coroutine_threadsafe(
                _put_nowait(signal_queue, signals), loop
            )

            # 이상 감지 시 프레임 배치를 VLM 파이프라인으로 전송 (얼굴 모자이크 처리 후)
            if len(frame_buffer) == 4 and (occlusion_active or body_sway_signal):
                if trace_data:
                    print(
                        f"[{zone_id}][DATA][BRIDGE->STATE] "
                        "sending 4-frame batch for VLM confirmation"
                    )
                mosaiced_batch = [apply_face_mosaic(f, face_detector_image) for f in frame_buffer]
                asyncio.run_coroutine_threadsafe(
                    _put_nowait(frame_queue, mosaiced_batch), loop
                )

    finally:
        cap.release()
        hands_detector.__exit__(None, None, None)
        if pose_detector is not None:
            pose_detector.__exit__(None, None, None)
        if face_detector_video is not None:
            face_detector_video.__exit__(None, None, None)
        if face_detector_image is not None:
            face_detector_image.__exit__(None, None, None)
        if face_landmarker_video is not None:
            face_landmarker_video.__exit__(None, None, None)
        if debug:
            cv2.destroyAllWindows()
        print(f"[{zone_id}][BRIDGE] Camera stopped")


async def _put_nowait(queue: asyncio.Queue, item) -> None:
    """큐가 꽉 찼으면 오래된 것을 버리고 최신 값으로 교체."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(item)


async def run(
    zone_id: str,
    frame_queue: asyncio.Queue,
    signal_queue: asyncio.Queue,
    camera_source: CameraSource = 0,
) -> None:
    """main.py에서 asyncio.create_task()로 호출하는 진입점."""
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(
        _run_detection_loop,
        zone_id, frame_queue, signal_queue, loop, camera_source,
    )
