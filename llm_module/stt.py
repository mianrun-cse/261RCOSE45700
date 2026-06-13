"""
공용 STT 유틸리티 — 마이크 녹음 + OpenAI Whisper 전사.

테스트 모듈에서 사용자가 직접 마이크에 말하면 한국어 텍스트로 변환해
customer_bot에 그대로 전달할 수 있게 한다.

의존성: sounddevice, soundfile (requirements.txt 참고)
"""
import asyncio
import os
import pathlib
import re
import tempfile

import numpy as np
from openai import OpenAI

STT_MODEL    = os.getenv("STT_MODEL", "whisper-1")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
SAMPLE_RATE  = 16000  # Whisper 권장 샘플레이트

# 무음 게이트 — 녹음 RMS가 이 값 이하이면 무음으로 보고 Whisper를 호출하지 않는다.
#   int16 기준. 환경(마이크 게인)에 따라 200~800 사이에서 조정.
STT_SILENCE_RMS = float(os.getenv("STT_SILENCE_RMS", "300"))

# Whisper 환각(무음·잡음 구간에서 나오는 유튜브 자막류 문구) 블록리스트.
#   공백/문장부호를 제거하고 부분 일치로 검사한다.
_HALLUCINATION_PHRASES = [
    "시청해주셔서감사합니다",
    "시청해주셔서감사드립니다",
    "구독과좋아요",
    "구독좋아요",
    "좋아요와구독",
    "다음영상에서만나요",
    "다음시간에만나요",
    "영상편집",
    "한글자막by",
    "엔딩",
    "이번영상은여기까지",
    "MBC뉴스",
    "감사합니다다음",
]


client = OpenAI()


def _compute_rms(frames: "np.ndarray") -> float:
    """int16 오디오 프레임의 RMS 에너지."""
    if frames is None or frames.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frames.astype(np.float64) ** 2)))


def _is_hallucination(text: str) -> bool:
    """전사 결과가 알려진 Whisper 환각 문구인지 판정."""
    compact = re.sub(r"[\s.,!?~…·\"']", "", text)
    if not compact:
        return True
    return any(p in compact for p in _HALLUCINATION_PHRASES)


def _record_blocking(seconds: float, samplerate: int) -> tuple[pathlib.Path, "numpy.ndarray"]:
    """sounddevice로 동기 녹음 → 임시 WAV 파일 저장."""
    import sounddevice as sd  # 지연 import (마이크 미사용 모드에서는 로드 안 되도록)
    import soundfile as sf

    frames = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    fd, name = tempfile.mkstemp(suffix=".wav", prefix="stt_")
    os.close(fd)
    path = pathlib.Path(name)
    sf.write(str(path), frames, samplerate)
    return path, frames


async def record_and_transcribe(seconds: float = 5.0) -> str:
    """
    `seconds`초 동안 마이크에서 녹음 → Whisper API로 한국어 전사.
    반환: 전사된 텍스트(strip). 인식 실패/무음이면 빈 문자열.
    """
    print(f"[STT] {seconds:.0f}초간 녹음합니다. 지금 말씀하세요...")
    path, frames = await asyncio.to_thread(_record_blocking, seconds, SAMPLE_RATE)

    # 1) 무음 게이트 — 에너지가 낮으면 환각을 유발하는 무음 전송을 막고 즉시 종료
    rms = _compute_rms(frames)
    if rms < STT_SILENCE_RMS:
        print(f"[STT] 무음 감지 (RMS={rms:.0f} < {STT_SILENCE_RMS:.0f}) — 전사 생략")
        try:
            path.unlink()
        except OSError:
            pass
        return ""

    print(f"[STT] 녹음 완료 (RMS={rms:.0f}). Whisper로 전사 중...")

    try:
        with path.open("rb") as f:
            result = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model=STT_MODEL,
                file=f,
                language=STT_LANGUAGE,
                temperature=0,  # 3) 환각 빈도 감소
            )
    finally:
        try:
            path.unlink()
        except OSError:
            pass

    text = (result.text or "").strip()

    # 2) 후처리 블록리스트 — 무음 게이트를 통과한 잡음에서 나온 환각 문구 제거
    if _is_hallucination(text):
        print(f"[STT] 환각 문구로 판단되어 폐기: {text!r}")
        return ""

    return text


def play_audio(path: str) -> None:
    """결과 음성을 스피커로 재생 (테스트 편의용). 실패해도 조용히 패스."""
    try:
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(path)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"[STT] 재생 실패(무시): {e}")
