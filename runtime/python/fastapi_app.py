from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from fastapi.responses import StreamingResponse
import io
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
    cosyvoice = CosyVoice2(
        model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False
    )
except Exception as e:
    # If initialization fails, log the error and handle it gracefully
    # For now, we'll let the app start but the endpoint will fail.
    cosyvoice = None
    print(f"Error initializing CosyVoice model: {e}")


class TTSRequest(BaseModel):
    text: str
    prompt: str
    wav: str
    speed: float = 1.0


class StreamTTSRequest(BaseModel):
    texts: List[str]  # 複数のテキストチャンク
    prompt: str
    wav: str
    speed: float = 1.0


async def common_tts_logic(
    texts, prompt: str, wav_path: str, speed: float = 1.0, stream: bool = False
):
    """
    共通のTTS処理ロジック

    Args:
        texts: テキスト（単一文字列またはイテレータ）
        prompt: プロンプトテキスト
        wav_path: 参照音声ファイルパス
        speed: 音声生成速度
        stream: ストリーミングモードかどうか

    Returns:
        stream=True: ジェネレーター（JSON形式）
        stream=False: 結合された音声データ
    """
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="CosyVoice model is not available.")

    try:
        prompt_speech_16k = load_wav(wav_path, 16000)

        # 音声生成ジェネレーターを作成
        model_output_generator = cosyvoice.inference_zero_shot(
            texts, prompt, prompt_speech_16k, speed=speed, stream=stream
        )

        if stream:
            # ストリーミングモード: JSON形式でチャンクを返す
            def generate_audio_stream():
                try:
                    for i, model_output in enumerate(model_output_generator):
                        output_wav = model_output["tts_speech"].numpy()

                        # 音声データをbase64エンコード
                        buffer = io.BytesIO()
                        import soundfile as sf

                        sf.write(
                            buffer, output_wav.T, cosyvoice.sample_rate, format="WAV"
                        )
                        buffer.seek(0)

                        import base64

                        audio_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

                        # JSON形式でチャンクを送信
                        chunk_data = {
                            "chunk_index": i,
                            "audio_data": audio_data,
                            "sample_rate": cosyvoice.sample_rate,
                            "status": "chunk",
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                    # 完了通知
                    final_data = {
                        "status": "completed",
                        "total_chunks": i + 1 if "i" in locals() else 0,
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"

                except Exception as e:
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return generate_audio_stream()

        else:
            # 非ストリーミングモード: 全チャンクを結合してWAVファイルを返す
            audio_chunks = []
            try:
                for model_output in model_output_generator:
                    output_wav = model_output["tts_speech"].numpy()
                    audio_chunks.append(output_wav)
            except Exception as gen_error:
                # ジェネレーター処理エラーの処理
                if not audio_chunks:
                    # チャンクが全く生成されなかった場合はエラー
                    raise HTTPException(
                        status_code=500,
                        detail=f"Audio generation failed: {str(gen_error)}",
                    )
                # 部分的に生成された場合は警告ログで継続
                print(f"Warning: Audio generation partially failed: {gen_error}")

            if not audio_chunks:
                raise HTTPException(
                    status_code=500, detail="No audio data was generated"
                )

            # 全チャンクを結合
            if len(audio_chunks) == 1:
                final_audio = audio_chunks[0]
            else:
                import numpy as np

                final_audio = np.concatenate(audio_chunks, axis=0)

            return final_audio

    except HTTPException:
        # HTTP例外はそのまま再発生
        raise
    except Exception as e:
        # その他の予期しないエラー
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    単一テキストのTTS処理
    改行なしの一文を音声合成してWAVファイルを直接返す
    """
    try:
        # 共通ロジックを呼び出し（非ストリーミングモード）
        final_audio = await common_tts_logic(
            request.text, request.prompt, request.wav, request.speed, stream=False
        )

        # WAVファイルとして返却
        buffer = io.BytesIO()
        import soundfile as sf

        sf.write(buffer, final_audio.T, cosyvoice.sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/tts-stream")
async def text_to_speech_stream(request: StreamTTSRequest):
    """
    複数テキストのストリーミングTTS処理
    改行ありのリストを逐次処理し、音声データをJSON形式でストリーミング返す
    """

    def text_generator():
        """要求されたテキストのジェネレーター"""
        for text in request.texts:
            print(f"******** Processing text chunk: {text}")
            yield text

    try:
        # 共通ロジックを呼び出し（ストリーミングモード）
        audio_stream_generator = await common_tts_logic(
            text_generator(), request.prompt, request.wav, request.speed, stream=True
        )

        return StreamingResponse(
            audio_stream_generator,
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "CosyVoice FastAPI server is running."}
