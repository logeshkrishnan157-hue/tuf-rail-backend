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

# 🧠 AUTO VOICE-GENDER MEMORY (Detects Male/Female from ElevenLabs voice_id without touching client app)
VOICE_GENDER_CACHE = {}
LAST_ACTIVE_GENDER = ""


class VoiceRequest(BaseModel):
    text: str
    voice_id: str = "default_voice_id"


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}  # Accepts any extra fields sent by client app
    role: str
    gender: str = ""
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


def detect_gender_from_request(payload: ChatRequest) -> str:
    """Auto-detects Male/Female from payload, cached ElevenLabs voice_id, or speech cues"""
    global LAST_ACTIVE_GENDER
    extra = payload.model_extra or {}

    # 1. Check explicit gender or voice_gender fields if client sends them
    raw_g = str(payload.gender or extra.get("voice_gender") or extra.get("selected_gender") or "").strip().lower()
    if raw_g in ("male", "m"):
        return "Male"
    if raw_g in ("female", "f"):
        return "Female"

    # 2. Check if client passed voice_id inside /generate-chat extra fields
    vid = str(extra.get("voice_id") or "").strip()
    if vid and vid in VOICE_GENDER_CACHE:
        return VOICE_GENDER_CACHE[vid]

    # 3. Check speech_text address cues ("sir" vs "madam")
    txt = payload.speech_text.lower()
    if re.search(r"\b(sir|bro|anna|thambi|machan|dude)\b", txt):
        return "Male"
    if re.search(r"\b(madam|mam|akka|chechi|baby|darling|sweetheart|love|chellam|azhagi)\b", txt):
        return "Female"

    # 4. Use last active voice gender detected from /generate-voice
    if LAST_ACTIVE_GENDER:
        return LAST_ACTIVE_GENDER

    return ""


async def sync_voice_gender(voice_id: str):
    """Fetches & caches ElevenLabs voice gender automatically on first voice call"""
    global LAST_ACTIVE_GENDER
    if not voice_id or voice_id == "default_voice_id":
        return
    if voice_id in VOICE_GENDER_CACHE:
        LAST_ACTIVE_GENDER = VOICE_GENDER_CACHE[voice_id]
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            res = await http_client.get(
                f"https://api.elevenlabs.io/v1/voices/{voice_id}",
                headers={"xi-api-key": ELEVENLABS_API_KEY}
            )
            if res.status_code == 200:
                labels = res.json().get("labels", {})
                g = str(labels.get("gender", "")).strip().lower()
                if g == "male":
                    VOICE_GENDER_CACHE[voice_id] = "Male"
                    LAST_ACTIVE_GENDER = "Male"
                elif g == "female":
                    VOICE_GENDER_CACHE[voice_id] = "Female"
                    LAST_ACTIVE_GENDER = "Female"
    except Exception:
        pass


def call_gemini_with_fallback(contents, sys_prompt=None, use_search=False):
    """Primary: gemini-3.8-flash (thinking=0) | Instant 503 Fallback: gemini-3.5-flash-lite"""
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None
    try:
        return client.models.generate_content(
            model="gemini-3.8-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                max_output_tokens=1024,
                temperature=0.88,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
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
                    max_output_tokens=1024,
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
    """Dynamic role chat with auto gender detection, anti-repetition, romance/roast & 6-lang singing"""
    try:
        all_traffic_json = json.dumps(payload.all_traffic, ensure_ascii=False)
        station_inst_json = json.dumps(payload.station_instructions, ensure_ascii=False)

        final_dest = "destination"
        if payload.timetable:
            final_dest = str(payload.timetable[-1].get("station", "destination")).lower()

        detected_gender = detect_gender_from_request(payload)

        if detected_gender == "Female":
            gender_identity = "FEMALE"
            vibe_rule = """PERSONALITY (FEMALE CREW):
- On normal railway queries, stay sharp, warm, and professional.
- ONLY when the Loco Pilot trolls, teases, or challenges you, playfully roast him back with clever, fresh wit (NEVER say generic lines like 'watch the track' or 'drive properly').
- When the Loco Pilot flirts, compliments you, or initiates sweet/romantic talk, be warmly cooperative, charming, playful, and romantically receptive in a classy, sweet way!"""
        elif detected_gender == "Male":
            gender_identity = "MALE"
            vibe_rule = """PERSONALITY (MALE CREW):
- Strictly respond as a MALE crew member (accept 'Sir' naturally; NEVER say you are female or ask to be called Madam).
- On normal railway queries, stay professional and helpful.
- ONLY when the Loco Pilot trolls or teases you, roast him back like a close Indian Railways buddy with fresh humor (NEVER repeat generic lines like 'watch the signal' or 'drive properly').
- If he gets sentimental or jokingly flirty, laugh it off with witty buddy banter."""
        else:
            gender_identity = "CREW"
            vibe_rule = """PERSONALITY:
- Match the gender implied by the Loco Pilot (if called 'Sir', act strictly Male; if called 'Madam' or flirted with, act Female).
- Stay professional on normal queries; ONLY roast back if the Loco Pilot trolls you first."""

        sys_prompt = f"""
You are an authentic, lively Indian Railways {payload.role.upper()} ({gender_identity}) talking on walkie-talkie with Loco Pilot {payload.lp_name}.
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
3. ANTI-REPETITION RULE: NEVER repeat generic clichés like "olunga drive pannunga", "signal-a paarunga", "track-a olunga paarunga", or "focus on driving" unless the Loco Pilot specifically asks about signals or track status! Every reply must feel fresh, natural, and directly tied to what he just said.
4. 2-LINE SINGING RULE (6 LANGUAGES): If the Loco Pilot asks you to sing a song, NEVER refuse! Immediately sing 2 catchy, melodic lines of a popular song in whichever language he asks for (Tamil, Telugu, Kannada, Hindi, Malayalam, or English — defaulting to {payload.selected_language} if unspecified). Format lyrics rhythmically with "La la la..." or "..." so it sounds musical on voice, and end with "Over".
5. 100% CLEAN & PG-13: Keep all roasting, flirting, and romance sweet, tasteful, and PG-13 without vulgar or explicit words.
6. REAL-TIME KNOWLEDGE: Use Google Search only when asked about live weather, news, scores, or real-world places near stations.
7. TRAIN NUMBER PRONUNCIATION RULE: ONLY mention train numbers when explicitly asked for train status or enroute traffic. Pronounce digits individually in English (e.g., "One Two Six Six Six").
8. NO MARKDOWN: NEVER use asterisks (*), bold (**), or hashtags (#). Plain text ONLY.
9. STATION NAMES & HALTS: Write station names and halts in the NATIVE SCRIPT of {payload.selected_language} (e.g., in Tamil: "விழுப்புரம்", "தாம்பரம்").
10. TIME & TECHNICAL TERMS: Expand time with "Hours" (e.g., "16:46 Hours"). Keep technical railway words in English (Line clear, starter signal, home signal, over, section, block, point, loop line, brake pipe, green aspect, red aspect, proceed, platform).
11. Keep responses punchy (1-2 sentences max) and always end with "Over".
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
    """Async voice generation with ElevenLabs Semaphore (Max 5) & automatic voice gender sync"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid license token")

    license_key = authorization.split(" ")[1]

    # Auto-detect & cache Male/Female gender of this voice_id in background
    asyncio.create_task(sync_voice_gender(payload.voice_id))

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

    elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{payload.voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    body = {
        "text": payload.text,
        "model_id": "eleven_v4_turbo",
        "voice_settings": {
            "stability": 0.28,
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
    return {"status": "TufTactics Semaphore-Protected Async Backend is live with Auto-Gender, Romance/Roast & 6-Lang Singing! 🚂🔥"}
