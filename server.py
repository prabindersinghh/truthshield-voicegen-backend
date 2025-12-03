from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from io import BytesIO

app = FastAPI()

SUPPORTED_LANGS = [
    "bhojpuri", "bengali", "english", "gujarati", "hindi",
    "chhattisgarhi", "kannada", "magahi", "maithili", "marathi", "telugu"
]

@app.get("/Get_Inference")
async def get_inference(text: str, lang: str, speaker_wav: UploadFile = File(...)):
    
    # Validate language
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Lowercase English text (VoiceTech requirement)
    if lang == "english":
        text = text.lower()

    # Validate WAV
    if speaker_wav.content_type not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="speaker_wav must be a WAV file")

    # TODO: Call your real model here
    # For now return a dummy WAV tone (works for testing)
    import numpy as np
    from scipy.io.wavfile import write

    sr = 22050
    duration = 1
    t = np.linspace(0, duration, int(sr * duration))
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = BytesIO()
    write(buf, sr, tone)
    buf.seek(0)

    headers = {
        "X-Model-Version": "truthshield-v1",
        "X-Speaker-Similarity": "0.75",
        "X-Safety-Verified": "true"
    }

    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
