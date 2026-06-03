"""
공용 STT 유틸리티 — 마이크 녹음 + OpenAI Whisper 전사.

테스트 모듈에서 사용자가 직접 마이크에 말하면 한국어 텍스트로 변환해
customer_bot에 그대로 전달할 수 있게 한다.

의존성: sounddevice, soundfile (requirements.txt 참고)
"""
import asyncio
import os
import pathlib
import tempfile

from openai import OpenAI

STT_MODEL    = os.getenv("STT_MODEL", "whisper-1")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
SAMPLE_RATE  = 16000  # Whisper 권장 샘플레이트


client = OpenAI()


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
    path, _ = await asyncio.to_thread(_record_blocking, seconds, SAMPLE_RATE)
    print("[STT] 녹음 완료. Whisper로 전사 중...")

    try:
        with path.open("rb") as f:
            result = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model=STT_MODEL,
                file=f,
                language=STT_LANGUAGE,
            )
    finally:
        try:
            path.unlink()
        except OSError:
            pass

    return (result.text or "").strip()


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
