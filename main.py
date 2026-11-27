import os
import re
import time
import asyncio
import traceback
from collections import defaultdict, deque
import httpx
import json
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Response, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()  # Loads keys from .env file

app = FastAPI(
    title="TufTactics Rail Assistant Safe-Concurrency Backend",
    docs_url=None,
    redoc_url=None
)

# Securely load credentials from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

# Initialize Supabase & GenAI Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# 🔒 SEMAPHORE GUARD: Restricts max concurrent calls to ElevenLabs matching your Creator Plan (5)
ELEVENLABS_SEMAPHORE = asyncio.Semaphore(5)

# 🛡️ RATE LIMIT & PER-USER GENDER MEMORY (Zero client changes needed)
RATE_LIMIT_STORE = defaultdict(deque)
VOICE_GENDER_CACHE = {}
IP_GENDER_CACHE = {}

MAX_AUDIO_BYTES = 3 * 1024 * 1024  # 3 MB max walkie-talkie audio
MAX_REQUESTS_PER_MIN = 18          # Max 18 PTT calls per minute per IP

DIGIT_MAP = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"
}


class VoiceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    voice_id: str = "default_voice_id"


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}  # Accepts any extra fields sent by client app
    role: str = Field(..., min_length=2, max_length=50)
    gender: str = ""
    speech_text: str = Field(..., min_length=1, max_length=600)
    lp_name: str = Field(..., min_length=1, max_length=80)
    train_name: str = Field(..., min_length=1, max_length=120)
    selected_language: str = Field(..., min_length=2, max_length=30)
    timetable: list
    all_traffic: list
    station_instructions: dict = {}
    communicated: list


class TopupRequest(BaseModel):
    email: str
    credits_to_add: int


def get_client_ip(request: Request) -> str:
    """Extracts real client IP behind Render proxy"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(client_ip: str, bucket: str):
    """Blocks automated API spam/drain without affecting normal gameplay"""
    now = time.time()
    key = f"{bucket}:{client_ip}"
    timestamps = RATE_LIMIT_STORE[key]
    while timestamps and now - timestamps[0] > 60:
        timestamps.popleft()
    if len(timestamps) >= MAX_REQUESTS_PER_MIN:
        raise HTTPException(status_code=429, detail="Too many radio transmissions. Please wait a few seconds.")
    timestamps.append(now)


def detect_gender_from_request(payload: ChatRequest, client_ip: str) -> str:
    """Auto-detects Male/Female per user IP without colliding across concurrent users"""
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

    # 4. Use last active voice gender detected for THIS specific user IP
    if client_ip in IP_GENDER_CACHE:
        return IP_GENDER_CACHE[client_ip]

    return ""


async def sync_voice_gender(voice_id: str, client_ip: str):
    """Fetches & caches ElevenLabs voice gender per user IP on first voice call"""
    if not voice_id or voice_id == "default_voice_id":
        return
    if voice_id in VOICE_GENDER_CACHE:
        IP_GENDER_CACHE[client_ip] = VOICE_GENDER_CACHE[voice_id]
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
                    IP_GENDER_CACHE[client_ip] = "Male"
                elif g == "female":
                    VOICE_GENDER_CACHE[voice_id] = "Female"
                    IP_GENDER_CACHE[client_ip] = "Female"
    except Exception:
        pass


def clean_tts_reply(raw_text: str) -> str:
    """Cleans markdown, brackets, adjacent duplicates & normalizes 5-digit train numbers for crisp TTS"""
    text = re.sub(r"\*+", "", raw_text)
    text = re.sub(r"#+", "", text)
    # Remove parenthetical English translations e.g. "விழுப்புரம் (Villupuram)"
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove adjacent duplicate words/phrases (1 to 3 words)
    text = re.sub(r"\b(\w+(?:\s+\w+){0,2})\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    # Convert raw 5-digit train numbers into comma-separated English digit words
    text = re.sub(r"\b\d{5}\b", lambda m: ", ".join(DIGIT_MAP[d] for d in m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()


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
async def transcribe_audio_from_client(request: Request, file: UploadFile = File(...)):
    """In-memory race-safe audio transcription with size guard & IP rate limit"""
    client_ip = get_client_ip(request)
    enforce_rate_limit(client_ip, "stt")

    try:
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Empty or corrupted audio clip.")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio clip exceeds 3MB limit.")

        stt_contents = [
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type="audio/wav",
            ),
            "Listen to this walkie-talkie transmission from a Loco Pilot. It can be in English, Tamil, Hindi, Telugu, Malayalam, Kannada, or mixed code-switching. Transcribe the exact words spoken in original script or phonetics. Output ONLY transcribed text."
        ]

        stt_response = await asyncio.to_thread(call_gemini_with_fallback, stt_contents, None, False)

        transcribed_text = stt_response.text.strip() if stt_response.text else "Signal status check, over."
        return {"text": transcribed_text}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"STT Transcription failed: {str(e)}")


@app.post("/generate-chat")
async def generate_role_chat_response(request: Request, payload: ChatRequest):
    """Dynamic role chat with IP rate limit, isolated gender memory & payload validation"""
    client_ip = get_client_ip(request)
    enforce_rate_limit(client_ip, "chat")

    if not payload.timetable or len(payload.timetable) > 200:
        raise HTTPException(status_code=400, detail="Invalid timetable context.")
    if len(payload.all_traffic) > 300:
        raise HTTPException(status_code=400, detail="Traffic payload too large.")

    try:
        all_traffic_json = json.dumps(payload.all_traffic, ensure_ascii=False)
        station_inst_json = json.dumps(payload.station_instructions, ensure_ascii=False)

        final_dest = "destination"
        if payload.timetable:
            final_dest = str(payload.timetable[-1].get("station", "destination")).lower()

        detected_gender = detect_gender_from_request(payload, client_ip)

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
4. 2-LINE SINGING RULE (6 LANGUAGES): If the Loco Pilot asks you to sing a song, NEVER refuse! Immediately sing 2 catchy, melodic lines of a popular song in whichever language he asks for (Tamil, Telugu, Kannada, Hindi, Malayalam, or English — defaulting to {payload.selected_language} if unspecified). Format lyrics rhythmically with "La la la..." or "..." so it sounds musical on voice. Do NOT say "Over" after singing.
5. 100% CLEAN & PG-13: Keep all roasting, flirting, and romance sweet, tasteful, and PG-13 without vulgar or explicit words.
6. REAL-TIME KNOWLEDGE: Use Google Search only when asked about live weather, news, scores, or real-world places near stations.
7. TRAIN NUMBER PRONUNCIATION RULE: ONLY mention train numbers when explicitly asked for train status or enroute traffic. Pronounce digits individually in English (e.g., "One, Two, Six, Six, Six").
8. NO MARKDOWN OR BRACKETS: NEVER use asterisks (*), bold (**), hashtags (#), or parenthetical translations in brackets like "விழுப்புரம் (Villupuram)". Plain text ONLY.
9. STATION NAMES & HALTS: Write station names and halts in the NATIVE SCRIPT of {payload.selected_language} (e.g., in Tamil: "விழுப்புரம்", "தாம்பரம்").
10. TIME & TECHNICAL TERMS: Expand time with "Hours" (e.g., "16:46 Hours"). Keep technical railway words in English (Line clear, starter signal, home signal, over, section, block, point, loop line, brake pipe, green aspect, red aspect, proceed, platform).
11. Keep responses punchy (1-2 sentences max). CONDITIONAL "OVER" RULE: End with "Over" ONLY when giving official railway/scenario communication (station clearance, signal aspect, timetable, halts, speed limits, traffic/crossing updates, brake/engine status, or crew duty orders). NEVER say "Over" during casual chat, jokes, roasting, flirting, singing, general knowledge, or weather/news replies.
"""

        user_msg = f"Loco Pilot {payload.lp_name}: {payload.speech_text}"

        response = await asyncio.to_thread(call_gemini_with_fallback, user_msg, sys_prompt, True)

        raw_reply = response.text if (response and response.text) else "Radio signal acknowledged, over."

        cleaned = clean_tts_reply(raw_reply)

        return {"reply": cleaned}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


@app.post("/generate-voice")
async def generate_voice_assistant_response(request: Request, payload: VoiceRequest, authorization: str = Header(None)):
    """Async voice generation with ElevenLabs Semaphore (Max 5) & per-IP voice gender sync"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid license token")

    client_ip = get_client_ip(request)
    license_key = authorization.split(" ")[1]

    # Auto-detect & cache Male/Female gender for this user IP in background
    asyncio.create_task(sync_voice_gender(payload.voice_id, client_ip))

    # 1. Verify License & Check Credits via Supabase RPC (Non-blocking)
    try:
        user_res = await asyncio.to_thread(
            lambda: supabase.rpc("verify_license_key", {"p_key": license_key.strip()}).execute()
        )
        if not user_res or not user_res.data:
            raise HTTPException(status_code=403, detail="Invalid License Key")

        user_data = user_res.data[0]
        user_email = user_data.get("user_email") or user_data.get("email")
        credits = int(user_data.get("credits", 0))
    except HTTPException:
        raise
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

    # 2. Deduct Credit Atomically via Supabase RPC (Non-blocking)
    try:
        await asyncio.to_thread(
            lambda: supabase.rpc("deduct_user_credit", {"p_email": user_email}).execute()
        )
    except Exception as e:
        print(f"[CREDIT DEDUCT ERROR]: {e}")

    return Response(content=tts_responses.content, media_type="audio/mpeg")


@app.post("/add-credits")
def add_user_credits_api(payload: TopupRequest, x_admin_secret: str = Header(None)):
    """Secure endpoint to add credits upon successful payment using Supabase RPC upsert safety"""
    if not ADMIN_SECRET_KEY or x_admin_secret != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized top-up request")

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
