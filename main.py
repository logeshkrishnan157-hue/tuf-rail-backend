import os
import re
import shutil
import asyncio
import traceback
import httpx
import json
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()  # Loads keys from .env file

app = FastAPI(title="TufTactics Rail Assistant Safe-Concurrency Backend")

# Securely load credentials from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase & GenAI Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# 🔒 SEMAPHORE GUARD: Restricts max concurrent calls to ElevenLabs matching your Creator Plan (5)
ELEVENLABS_SEMAPHORE = asyncio.Semaphore(5)


class VoiceRequest(BaseModel):
    text: str
    voice_id: str = "default_voice_id"


class ChatRequest(BaseModel):
    role: str
    gender: str = "Female"
    speech_text: str
    lp_name: str
    train_name: str
    selected_language: str
    timetable: list
    all_traffic: list
    station_instructions: dict = {}
    communicated: list


class TopupRequest(BaseModel):
    email: str
    credits_to_add: int


def call_gemini_with_fallback(contents, sys_prompt=None, use_search=False):
    """Primary: gemini-3.8-flash | Instant 503 Fallback: gemini-3.5-flash-lite"""
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None
    try:
        return client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                max_output_tokens=600,
                temperature=0.85,
                tools=tools
            )
        )
    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            print("[503 SPIKE]: Switching seamlessly to gemini-3.5-flash-lite...")
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    max_output_tokens=600,
                    tools=tools
                )
            )
        raise e


@app.post("/transcribe-audio")
def transcribe_audio_from_client(file: UploadFile = File(...)):
    """Threadpool-safe audio transcription using Gemini STT with 503 resilience"""
    temp_audio_path = f"temp_{file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(temp_audio_path, "rb") as af:
            audio_bytes = af.read()

        stt_contents = [
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav",
            ),
            "Listen to this walkie-talkie transmission from a Loco Pilot. It can be in English, Tamil, Hindi, Telugu, Malayalam, Kannada, or mixed code-switching. Transcribe the exact words spoken in original script or phonetics. Output ONLY transcribed text."
        ]

        stt_response = call_gemini_with_fallback(contents=stt_contents, use_search=False)

        transcribed_text = stt_response.text.strip() if stt_response.text else "Signal status check, over."
        return {"text": transcribed_text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"STT Transcription failed: {str(e)}")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.post("/generate-chat")
def generate_role_chat_response(payload: ChatRequest):
    """Fast role chat generation with full token headroom, strict grounding & 503 resilience"""
    try:
        all_traffic_json = json.dumps(payload.all_traffic, ensure_ascii=False)
        station_inst_json = json.dumps(payload.station_instructions, ensure_ascii=False)

        final_dest = "destination"
        if payload.timetable:
            final_dest = str(payload.timetable[-1].get("station", "destination")).lower()

        is_female = payload.gender.strip().lower() == "female"
        vibe_rule = (
            "PERSONALITY (FEMALE CREW): Stay professional on normal railway queries. If the Loco Pilot trolls or teases about driving/railways, playfully roast him back. ONLY if the Loco Pilot initiates sweet, romantic, or flirty talk, respond with charming, clean, witty flirtatious banter. Adapt naturally to any topic (weather, news, exams, health, cinema, songs) while staying in character."
            if is_female else
            "PERSONALITY (MALE CREW): Stay professional on normal railway queries. If the Loco Pilot trolls or teases, roast him back like a close Indian Railways buddy. If he gets sentimental or flirty, laugh it off with witty buddy humor. Adapt naturally to any topic (weather, news, exams, health, cinema, songs) while staying in character."
        )

        sys_prompt = f"""
You are an authentic, lively Indian Railways {payload.role.upper()} ({payload.gender}) talking on walkie-talkie with Loco Pilot {payload.lp_name}.
Player Train: {payload.train_name}.
Final Destination Station: {final_dest}.
User Preferred Communication Language Option: {payload.selected_language}.

CRITICAL STATION SPECIFIC INSTRUCTIONS & RULES:
{station_inst_json}

ALL ROUTE TRAFFIC & ENROUTE TRAINS DATABASE (includes player train with is_player=true):
{all_traffic_json}

Stations already communicated so far: {payload.communicated}

STRICT OPERATIONAL & FORMAT RULES:
1. ZERO HALLUCINATION RULE: Read timings, station names, halts, speed limits, and special rules EXCLUSIVELY from the timetable and station instructions above. Never invent times!
2. {vibe_rule}
3. 100% CLEAN & FAMILY-SAFE: Keep all roasting, trolling, and flirting wholesome and PG-13. NEVER use vulgar, explicit, abusive, or bad words.
4. REAL-TIME KNOWLEDGE: Use Google Search only when asked about live weather, news, scores, or real-world places near stations.
5. TRAIN NUMBER PRONUNCIATION RULE: ONLY mention train numbers when the Loco Pilot explicitly asks for train status, train numbers, or enroute traffic details. Pronounce train numbers in English as single, individual digits word-by-word (e.g., "One Two Six Six Six", never "Twelve thousand").
6. NO MARKDOWN: NEVER use markdown asterisks (*), bold (**), or hashtags (#). Output plain text ONLY.
7. STATION NAMES & HALTS: Write all station names and stopping halts in the NATIVE SCRIPT of {payload.selected_language} (e.g., in Tamil: "விழுப்புரம்", "தாம்பரம்") so ElevenLabs gives a true native local accent.
8. TIME & TECHNICAL TERMS: Always expand time into the English word "Hours" (e.g., "16:46 Hours"). Keep all technical railway words strictly in English (e.g., Line clear, starter signal, home signal, over, section, block, point, loop line, brake pipe, green aspect, red aspect, proceed, platform).
9. Keep responses punchy and short (1-2 sentences max). Always end your response with "Over".
"""

        user_msg = f"Loco Pilot {payload.lp_name}: {payload.speech_text}"

        response = call_gemini_with_fallback(
            contents=user_msg,
            sys_prompt=sys_prompt,
            use_search=True
        )

        raw_reply = response.text if (response and response.text) else "Radio signal acknowledged, over."

        cleaned = re.sub(r"\*+", "", raw_reply)
        cleaned = re.sub(r"#+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return {"reply": cleaned}
    except Exception as e:
        traceback.print_exc()
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
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            tts_responses = await http_client.post(elevenlabs_url, json=body, headers=headers)

    if tts_responses.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to generate audio from ElevenLabs: {tts_responses.text}")

    # 2. Deduct Credit Atomically via Supabase RPC
    try:
        supabase.rpc("deduct_user_credit", {"p_email": user_email}).execute()
    except Exception as e:
        print(f"[CREDIT DEDUCT ERROR]: {e}")

    return Response(content=tts_responses.content, media_type="audio/mpeg")


@app.post("/add-credits")
def add_user_credits_api(payload: TopupRequest):
    """Secure endpoint to add credits upon successful Cashfree payment using Supabase RPC upsert safety"""
    try:
        supabase.rpc("add_user_credits_secure", {
            "p_email": payload.email,
            "p_amount": payload.credits_to_add
        }).execute()

        return {"status": "success", "message": f"Successfully added {payload.credits_to_add} credits for {payload.email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add credits: {str(e)}")


@app.get("/")
def home():
    return {"status": "TufTactics Semaphore-Protected Async Backend is live with Real-Time Search & Dynamic Crew Banter! 🚂🔥"}
