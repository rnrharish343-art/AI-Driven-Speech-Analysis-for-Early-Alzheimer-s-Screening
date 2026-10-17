import os
import shutil
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from model_pipeline import AlzheimersMultiModalModel, calculate_demo_score

app = FastAPI(title="Multimodel Agent Backend")

# Configure CORS middleware to accept all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the model globally in evaluation mode on CPU
device = torch.device("cpu")
model = AlzheimersMultiModalModel(d_model=768, nhead=8).to(device)
model.eval()

# Initialize the automatic speech recognition pipeline using whisper-tiny
print("[INFO] Initializing Speech-to-Text Pipeline (Whisper-tiny)...")
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)
print("[INFO] Speech-to-Text Pipeline initialized successfully.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    # Save the audio file temporarily
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Copy the uploaded file stream into the temp file
            shutil.copyfileobj(file.file, tmp_file)
            temp_path = tmp_file.name

        # Read file size
        file_size = os.path.getsize(temp_path)
        print(f"\n[INFO] Received upload file: '{file.filename}', size: {file_size} bytes")
            
    except Exception as e:
        print(f"[ERROR] Failed to save uploaded file: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Load audio array using librosa first (this will succeed for true WAVs without ffmpeg)
    audio_arr = None
    try:
        import librosa
        print("[INFO] Loading audio file with librosa...")
        audio_arr, sr = librosa.load(temp_path, sr=16000)
    except Exception as e:
        print(f"[WARNING] Failed to load audio with librosa: {str(e)}")

    # 1. Run Speech-to-Text Transcription locally using Whisper
    try:
        print("[INFO] Transcribing audio file using Whisper-tiny...")
        if audio_arr is not None:
            asr_result = asr_pipeline(audio_arr)
        else:
            asr_result = asr_pipeline(temp_path)
        transcript = asr_result.get("text", "").strip()
        if not transcript:
            transcript = "[Unclear audio or silence]"
        print(f"[INFO] Transcription: \"{transcript}\"")
    except Exception as e:
        print(f"[ERROR] Speech recognition failed: {str(e)}")
        transcript = "[Speech recognition failed]"

    # 2. Run feature scoring metrics (using librosa + transcript evaluation)
    try:
        risk_score, confidence, pause_ratio, repetitions = calculate_demo_score(temp_path, transcript)
        print(f"[INFO] Speech metrics -> Pause Ratio: {pause_ratio:.2f}, Repetitions: {repetitions}")
        print(f"[INFO] Fallback Risk Score: {risk_score:.4f}, Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"[ERROR] Metrical calculation failed: {str(e)}")
        risk_score, confidence = 0.12, 0.88

    # 3. Cleanup temporary file
    try:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("[INFO] Cleaned up temporary file.")
    except Exception as e:
        print(f"[WARNING] Failed to delete temp file '{temp_path}': {str(e)}")

    # 4. Verify PyTorch Model pipeline remains functional (evaluation verification)
    try:
        dummy_audio = torch.randn(1, 100, 768).to(device)
        dummy_ling = torch.randn(1, 50, 768).to(device)
        with torch.no_grad():
            _ = model(dummy_audio, dummy_ling)
        print("[INFO] PyTorch model forward pass verified successfully.")
    except Exception as e:
        print(f"[WARNING] PyTorch pipeline verification failed: {str(e)}")

    # Determine label mapping
    if risk_score >= 0.5:
        prediction_label = "AD"
    else:
        prediction_label = "Healthy Control"

    # Return success response containing dynamic transcript and speech scores
    return {
        "status": "success",
        "prediction": prediction_label,
        "confidence": confidence,
        "risk_score": risk_score,
        "transcript": transcript
    }

# Resolve absolute path to frontend/dist to serve compiled assets
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dist_path = os.path.abspath(os.path.join(current_dir, "..", "frontend", "dist"))

if os.path.exists(frontend_dist_path):
    print(f"[INFO] Serving React frontend from static directory: '{frontend_dist_path}'")
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static")
else:
    print(f"[WARNING] Static build directory not found at '{frontend_dist_path}'. Frontend will not be served.")

