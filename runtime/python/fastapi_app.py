from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from fastapi.responses import StreamingResponse
import io
import pykakasi

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


@app.get("/")
def read_root():
    return {"message": "CosyVoice FastAPI server is running."}
