"""공용 TTS 유틸리티. 텍스트를 MP3 음성 파일로 변환한다."""
import os
import asyncio
import pathlib
from openai import OpenAI

client = OpenAI()

TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "nova")
AUDIO_DIR = pathlib.Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)


async def _tts(text: str, filename: str) -> pathlib.Path:
    """텍스트 → MP3 파일 생성"""
    path = AUDIO_DIR / filename
    response = await asyncio.to_thread(
        client.audio.speech.create,
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    response.stream_to_file(str(path))
    return path
