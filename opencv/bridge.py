"""
face_detection.py의 occlusion_active 신호를 state_machine 큐에 연결하는 어댑터.

USB 웹캠(int) 또는 Tapo RTSP URL(str)을 지원한다.
face_detection.py는 독립 실행 스크립트이므로 직접 import 대신
같은 라이브러리(MediaPipe, OpenCV)를 사용해 핵심 로직만 재활용한다.
"""
import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import asyncio
import time
from typing import Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from llm_module.state_machine import TriggerSignals

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, 'hand_landmarker.task')
FRAME_SIZE = (640, 360)
OCCLUSION_TRIGGER_FRAMES = 10
PERSON_GONE_TRIGGER_FRAMES = 30
SAMPLE_INTERVAL_SEC = 0.4

CameraSource = Union[int, str]


def _open_capture(camera_source: CameraSource) -> cv2.VideoCapture:
    """Open USB webcam index or RTSP URL."""
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
    """face_detection.py의 hand_near_face() 그대로 재사용"""
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


def _run_detection_loop(
    zone_id: str,
    frame_queue: asyncio.Queue,
    signal_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    camera_source: CameraSource = 0,
) -> None:
    """
    동기 OpenCV 루프. 별도 스레드에서 실행된다.
    감지 결과를 asyncio 큐에 thread-safe하게 넣는다.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[{zone_id}][BRIDGE] MediaPipe model not found: {MODEL_PATH}")
        print(f"[{zone_id}][BRIDGE] Download hand_landmarker.task to the project root.")
        return

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)
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

    last_face_region = None
    last_known_customer = None
    occlusion_frame_count = 0
    occlusion_active = False
    person_gone_count = 0

    frame_buffer = []
    last_sample_time = 0.0
    frame_index = 0
    last_data_log_time = 0.0

    debug = os.getenv("DEBUG_CAMERA", "1") == "1"
    trace_data = os.getenv("TRACE_CAMERA_DATA", "1") == "1"
    use_rtsp = isinstance(camera_source, str)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{zone_id}][BRIDGE] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            if use_rtsp or frame.shape[1] != FRAME_SIZE[0] or frame.shape[0] != FRAME_SIZE[1]:
                frame = cv2.resize(frame, FRAME_SIZE)

            now = time.time()
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = frame_index * 33
            frame_index += 1

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            hand_results = hands_detector.detect_for_video(mp_image, timestamp_ms)

            current_customer = "customer" if len(faces) > 0 else None

            if current_customer and len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                last_face_region = (fx, fy, fw, fh)
                last_known_customer = current_customer

            known_missing = last_known_customer is not None and current_customer is None

            if known_missing and last_face_region is not None:
                hand_detected = _hand_near_face(hand_results, last_face_region, frame.shape)
                if hand_detected:
                    occlusion_frame_count += 1
                    person_gone_count = 0
                    if occlusion_frame_count >= OCCLUSION_TRIGGER_FRAMES:
                        occlusion_active = True
                else:
                    occlusion_frame_count = max(0, occlusion_frame_count - 1)
                    person_gone_count += 1
                    if person_gone_count >= PERSON_GONE_TRIGGER_FRAMES:
                        occlusion_active = False
                        last_known_customer = None
                        last_face_region = None
                        occlusion_frame_count = 0
                        person_gone_count = 0
            elif current_customer is not None:
                occlusion_frame_count = 0
                occlusion_active = False
                person_gone_count = 0

            if debug:
                h_img, w_img = frame.shape[:2]
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if hand_results.hand_landmarks:
                    for hand_lms in hand_results.hand_landmarks:
                        for lm in hand_lms:
                            cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                            cv2.circle(display, (cx, cy), 4, (0, 215, 255), -1)
                status = "SWEAT DETECTED!" if occlusion_active else "monitoring..."
                color = (0, 0, 255) if occlusion_active else (0, 255, 0)
                cv2.putText(display, status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(display, f"occlusion: {occlusion_frame_count}/{OCCLUSION_TRIGGER_FRAMES}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"faces: {len(faces)}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow(f"Bridge - {zone_id}", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if now - last_sample_time >= SAMPLE_INTERVAL_SEC:
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > 4:
                    frame_buffer.pop(0)
                last_sample_time = now

            signals = TriggerSignals(
                sweat_wiping=occlusion_active,
                person_count=len(faces),
            )

            if trace_data and now - last_data_log_time >= 1.0:
                hand_count = len(hand_results.hand_landmarks or [])
                print(
                    f"[{zone_id}][DATA][BRIDGE->STATE] "
                    f"faces={len(faces)} hands={hand_count} "
                    f"occlusion_count={occlusion_frame_count}/{OCCLUSION_TRIGGER_FRAMES} "
                    f"sweat_wiping={signals.sweat_wiping} "
                    f"person_count={signals.person_count} "
                    f"frame_buffer={len(frame_buffer)}/4"
                )
                last_data_log_time = now

            asyncio.run_coroutine_threadsafe(
                _put_nowait(signal_queue, signals), loop
            )

            if len(frame_buffer) == 4 and occlusion_active:
                if trace_data:
                    print(
                        f"[{zone_id}][DATA][BRIDGE->STATE] "
                        "sending 4-frame batch for VLM confirmation"
                    )
                asyncio.run_coroutine_threadsafe(
                    _put_nowait(frame_queue, list(frame_buffer)), loop
                )

    finally:
        cap.release()
        hands_detector.__exit__(None, None, None)
        if debug:
            cv2.destroyAllWindows()
        print(f"[{zone_id}][BRIDGE] Camera stopped")


async def _put_nowait(queue: asyncio.Queue, item) -> None:
    """큐가 꽉 찼으면 오래된 것을 버리고 최신 값으로 교체"""
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
    """main.py에서 asyncio.create_task()로 호출하는 진입점"""
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(
        _run_detection_loop,
        zone_id, frame_queue, signal_queue, loop, camera_source,
    )
