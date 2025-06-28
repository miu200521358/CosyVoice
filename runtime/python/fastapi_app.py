from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

# Initialize the CosyVoice model
# Adjust the model path and other parameters as needed
model_dir = "/opt/CosyVoice/pretrained_models/CosyVoice-300M"
try:
    cosyvoice = CosyVoice(model_dir=model_dir)
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


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    if cosyvoice is None:
        raise HTTPException(status_code=500, detail="CosyVoice model is not available.")

    try:
        prompt_speech_16k = load_wav(request.wav, 16000)

        # Generate speech
        model_output_generator = cosyvoice.inference_zero_shot(
            request.text, request.prompt, prompt_speech_16k
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
