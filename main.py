import os
import shutil
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai

load_dotenv() # Loads keys from .env file

app = FastAPI(title="TufTactics Rail Assistant Safe-Concurrency Backend")

# Securely load credentials from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase & Gemini
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3.8-flash')

# 🔒 SEMAPHORE GUARD: Restricts max concurrent calls to ElevenLabs matching your Creator Plan (5)
ELEVENLABS_SEMAPHORE = asyncio.Semaphore(5)

class VoiceRequest(BaseModel):
    text: str
    voice_id: str = "default_voice_id"

class ChatRequest(BaseModel):
    role: str
    speech_text: str
    lp_name: str
    train_name: str
    selected_language: str
    timetable: list
    all_traffic: list
    communicated: list

@app.post("/transcribe-audio")
async def transcribe_audio_from_client(file: UploadFile = File(...)):
    """Non-blocking audio transcription using Gemini STT"""
    temp_audio_path = f"temp_{file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        with open(temp_audio_path, "rb") as af:
            audio_bytes = af.read()

        stt_response = model.generate_content([
            {
                "mime_type": "audio/wav",
                "data": audio_bytes
            },
            "Listen to this walkie-talkie transmission from a Loco Pilot. It can be in English, Tamil, Hindi, Telugu, Malayalam, Kannada, or mixed code-switching. Transcribe the exact words spoken in original script or phonetics. Output ONLY transcribed text."
        ])
        
        transcribed_text = stt_response.text.strip() if stt_response.text else "Signal status check, over."
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return {"text": transcribed_text}
    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"STT Transcription failed: {str(e)}")

@app.post("/generate-chat")
async def generate_role_chat_response(payload: ChatRequest):
    """Asynchronous role chat generation"""
    try:
        player_timetable_json = str(payload.timetable)
        all_traffic_json = str(payload.all_traffic)
        
        final_dest = "destination"
        if payload.timetable:
            final_dest = str(payload.timetable[-1].get("station", "destination")).lower()

        sys_prompt = f"""
        You are an authentic Indian Railways {payload.role.upper()}. Loco Pilot name is {payload.lp_name}.
        Player Train: {payload.train_name}.
        Final Destination Station: {final_dest}.
        User Preferred Communication Language Option: {payload.selected_language}.
        
        CRITICAL GROUNDING DATABASE (DO NOT DEVIATE):
        {player_timetable_json}
        
        ALL ROUTE TRAFFIC & ENROUTE TRAINS DATABASE:
        {all_traffic_json}
        
        Stations already communicated so far: {payload.communicated}

        CRITICAL INSTRUCTIONS FOR NATURAL PRONUNCIATION & ACCURACY:
        1. **ZERO HALLUCINATION RULE:** Read timings and station names EXCLUSIVELY from the "CRITICAL GROUNDING DATABASE". Never invent times!
        2. **TRAIN NUMBER PRONUNCIATION RULE:** When mentioning any train number (e.g., 12605), you MUST pronounce it in English as **single, individual digits** word-by-word (e.g., "One Two Six Zero Five"). Never read train numbers as combined numbers like "Twelve thousand"!
        3. **TRAIN NUMBER CONTEXT:** ONLY mention train numbers when the Loco Pilot explicitly asks for train status, train numbers, or enroute traffic details. Do NOT repeat the train number for standard clearance, signals, or timetable halts.
        4. **NO MARKDOWN:** NEVER use markdown asterisks (*), bold (**), or hashtags (#). Output plain text ONLY.
        5. **STATION NAMES & HALTS:** Print all station names and stopping halts in the **NATIVE SCRIPT** of the selected regional language (e.g., in Tamil: "தாம்பரம்", "செங்கல்பட்டு") so ElevenLabs gives a true native local accent.
        6. **TIME & HRS:** Always expand time into the English word **"Hours"** (e.g. "16:07 Hours").
        7. **TECHNICAL RAILWAY TERMS:** Keep all technical railway words strictly in English (e.g., Line clear, starter signal, over, section, block, point, loop line, brake pipe, green aspect, red aspect, proceed, platform).
        8. Keep responses authoritative, professional, and short (1-2 sentences max). Always end your response with 'Over'.
        """

        response = model.generate_content([sys_prompt, f"Loco Pilot {payload.lp_name}: {payload.speech_text}"])
        raw_reply = response.text if response.text else "Radio signal acknowledged, over."
        
        import re
        cleaned = re.sub(r'\*+', '', raw_reply)
        cleaned = re.sub(r'#+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return {"reply": cleaned}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

@app.post("/generate-voice")
async def generate_voice_assistant_response(payload: VoiceRequest, authorization: str = Header(None)):
    """Async voice generation protected with ElevenLabs Concurrency Semaphore (Max 5)"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid license token")
    
    license_key = authorization.split(" ")[1]

    # 1. Verify License & Check Credits via Supabase RPC
    try:
        user_res = supabase.rpc("verify_license_key", {"p_key": license_key.strip()}).execute()
        if not user_res or not user_res.data:
            raise HTTPException(status_code=403, detail="Invalid License Key")
        
        user_data = user_res.data[0]
        user_email = user_data.get("user_email") or user_data.get("email")
        credits = int(user_data.get("credits", 0))
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"License verification failed: {str(e)}")

    if credits <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits. Please top-up on TufScenarios hub.")

    ai_response_text = payload.text 

    elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{payload.voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    body = {
        "text": ai_response_text,
        "model_id": "eleven_v4_turbo",
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.85
        }
    }

    # 🔒 ACQUIRE SEMAPHORE: Ensures maximum 5 concurrent calls hit ElevenLabs at any exact microsecond
    async with ELEVENLABS_SEMAPHORE:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tts_responses = await client.post(elevenlabs_url, json=body, headers=headers)
    
    if tts_responses.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to generate audio from ElevenLabs: {tts_responses.text}")

    # 2. Deduct Credit Atomically via Supabase RPC
    try:
        supabase.rpc("deduct_user_credit", {"p_email": user_email}).execute()
    except Exception as e:
        print(f"[CREDIT DEDUCT ERROR]: {e}")

    from fastapi.responses import Response
    return Response(content=tts_responses.content, media_type="audio/mpeg")

@app.get("/")
def home():
    return {"status": "TufTactics Semaphore-Protected Async Backend is live! 🚂🔥"}
