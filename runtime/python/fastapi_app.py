from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from fastapi.responses import StreamingResponse
import io
import pykakasi
import json
import torchaudio
import tempfile
import os

app = FastAPI()

# Initialize the CosyVoice model
# Adjust the model path and other parameters as needed
# model_dir = "/opt/CosyVoice/pretrained_models/CosyVoice-300M"
model_dir = "/opt/CosyVoice/pretrained_models/CosyVoice2-0.5B"
try:
    # cosyvoice = CosyVoice(model_dir)
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
except Exception as e:
    # If initialization fails, log the error and handle it gracefully
    # For now, we'll let the app start but the endpoint will fail.
    cosyvoice = None
    print(f"Error initializing CosyVoice model: {e}")


class TTSRequest(BaseModel):
    tts: str
    prompt: str
    wav: str
    speed: float = 1.0


class StreamTTSRequest(BaseModel):
    texts: List[str]  # 複数のテキストチャンク
    prompt: str
    wav: str
    speed: float = 1.0


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="CosyVoice model is not available.")

    try:
        prompt_speech_16k = load_wav(request.wav, 16000)

        kks = pykakasi.kakasi()
        result = kks.convert(request.tts)
        jp_tts = " ".join([item['hira'] for item in result])

        # Generate speech
        model_output_generator = cosyvoice.inference_zero_shot(
            jp_tts, request.prompt, prompt_speech_16k, speed=request.speed,
        )
        model_output = next(model_output_generator)
        output_wav = model_output["tts_speech"].numpy()

        # Convert the wav to a byte stream to be returned
        buffer = io.BytesIO()
        import soundfile as sf

        sf.write(buffer, output_wav.T, cosyvoice.sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts-stream")
async def text_to_speech_stream(request: StreamTTSRequest):
    """
    ストリーミングTTSエンドポイント
    複数のテキストチャンクを逐次処理し、音声データをJSON形式でストリーミング返す
    """
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="CosyVoice model is not available.")

    try:
        prompt_speech_16k = load_wav(request.wav, 16000)
        kks = pykakasi.kakasi()

        def text_generator():
            """テキストジェネレーター"""
            for text in request.texts:
                if text.strip():  # 空でないテキストのみ処理
                    result = kks.convert(text.strip())
                    jp_tts = " ".join([item['hira'] for item in result])
                    yield jp_tts

        def generate_audio_stream():
            """音声ストリーミングジェネレーター"""
            try:
                # CosyVoiceのinference_zero_shotでストリーミング処理
                model_output_generator = cosyvoice.inference_zero_shot(
                    text_generator(), 
                    request.prompt, 
                    prompt_speech_16k, 
                    speed=request.speed
                )

                for i, model_output in enumerate(model_output_generator):
                    output_wav = model_output["tts_speech"].numpy()

                    # 音声データをbase64エンコード
                    buffer = io.BytesIO()
                    import soundfile as sf
                    sf.write(buffer, output_wav.T, cosyvoice.sample_rate, format="WAV")
                    buffer.seek(0)
                    
                    import base64
                    audio_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    # JSON形式でチャンクを送信
                    chunk_data = {
                        "chunk_index": i,
                        "audio_data": audio_data,
                        "sample_rate": cosyvoice.sample_rate,
                        "status": "chunk"
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # 完了通知
                final_data = {
                    "status": "completed",
                    "total_chunks": i + 1 if 'i' in locals() else 0
                }
                yield f"data: {json.dumps(final_data)}\n\n"

            except Exception as e:
                error_data = {
                    "status": "error",
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_audio_stream(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "CosyVoice FastAPI server is running."}
