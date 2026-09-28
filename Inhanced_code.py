# Inhanced_code.py
import os
import re
import time
import logging
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from openai import OpenAI
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Check required env vars at startup
_REQUIRED_ENV = ["OPENROUTER_API_KEY", "MONGODB_URI"]
_missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
if _missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(_missing)}")

# App settings
MAX_HISTORY       = 10
MAX_MSG_LENGTH    = 500
RATE_LIMIT_MAX    = 30
RATE_LIMIT_WINDOW = 60

# OpenRouter client
openai_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# MongoDB collections
_mongo_client = MongoClient(os.getenv("MONGODB_URI"), serverSelectionTimeoutMS=5000)

try:
    _mongo_client.admin.command("ping")
    log.info("MongoDB connected.")
except (ConnectionFailure, OperationFailure) as e:
    raise ConnectionError(f"Could not connect to MongoDB: {e}")

_db            = _mongo_client[os.getenv("MONGODB_DB", "mwg_chatbot")]
_knowledge_col = _db["knowledge_chunks"]
_history_col   = _db["chatsession"]
_rate_col      = _db["rate_limits"]

_history_col.create_index([("user_id", ASCENDING)], background=True)
_rate_col.create_index(   [("user_id", ASCENDING)], background=True)
_rate_col.create_index(   [("window_start", ASCENDING)], expireAfterSeconds=120, background=True)

# Intent keywords
BOOKING_WORDS = [
    "book", "schedule", "reserve", "appointment", "slot", "wash now", "set up", "arrange",
    "gaadi wash", "car saaf", "gaadi saaf", "wash karna", "wash karwa",
    "booking karo", "book karo", "wash chahiye", "lagao", "karwa do",
    "karna hai", "karwana", "chahiye wash",
]

PREDICT_WORDS = [
    "brake", "tyre", "tire", "battery", "oil", "filter", "replace", "fail",
    "noise", "worn", "spark", "when should", "predict", "car health",
    "tyre kharab", "brake fail", "oil change", "battery dead",
]

FRANCHISE_WORDS = [
    "franchise", "start business", "open branch", "invest in mwg", "mwg franchise",
    "franchise kya hai", "franchise kaise", "franchise lena",
    "business kaise kare", "kitna invest", "1.9 lakh", "1,90,000", "190000",
]

FREELANCER_WORDS = [
    "freelancer", "earn money", "want to earn", "join as", "part time", "gig",
    "kamai", "rojgar", "paise kamao", "kaam chahiye", "job chahiye",
]

BYOB_WORDS = [
    "byob", "be your own boss", "own boss", "backpack kit", "gig worker",
    "boss bano", "apna boss", "khud ka boss", "24999", "24,999",
    "apna kaam", "khud ka kaam", "boss khud bano",
]

AREA_PARTNER_WORDS = [
    "area partner", "local partner", "society partner", "society subscription",
    "apna area", "area me kaam", "residential partner",
    "area partner kaise", "partner kaise bane", "apni society",
    "3000", "5000", "3,000", "5,000",
]

SUBSCRIPTION_WORDS = [
    "monthly plan", "yearly plan", "subscription", "membership",
    "annual plan", "monthly wash", "1299", "9999",
    "mahine ka plan", "saal ka plan", "monthly package", "yearly package",
    "subscribe karna", "plan lena",
]

BROAD_PHRASES = [
    "tell me everything", "all services", "all features",
    "what do you provide", "complete list", "everything you offer",
    "sab kuch batao", "poora bata do", "kya kya hai",
]

JOB_BDA_WORDS = [
    "business development", "bda", "bda job",
    "business development associate", "bd associate", "business associate",
    "office job mwg", "corporate job mwg",
    "2.4 lpa", "5.8 lpa", "1 lakh incentive", "jamshedpur job",
    "mwg job office", "work from office mwg",
]

JOB_FSE_WORDS = [
    "field sales", "fse", "sales executive", "field executive",
    "field job", "field sales executive", "sales job mwg",
    "society sales", "door to door", "hyperlocal sales",
    "15000 salary", "25000 salary", "8000 salary", "10000 salary",
    "field work job", "no degree job", "freshers job mwg",
    "mwg sales job", "car sales job",
]

# WhatsApp templates
TEMPLATES = {
    "booking": {
        "type": "template", "intent": "booking",
        "template": {
            "name": "mwg_booking_options", "language": {"code": "en"},
            "components": [
                {"type": "body", "parameters": []},
                {"type": "button", "sub_type": "quick_reply", "index": "0", "parameters": [{"type": "payload", "payload": "BOOK_BASIC"}]},
                {"type": "button", "sub_type": "quick_reply", "index": "1", "parameters": [{"type": "payload", "payload": "BOOK_PREMIUM"}]},
                {"type": "button", "sub_type": "quick_reply", "index": "2", "parameters": [{"type": "payload", "payload": "BOOK_DETAILING"}]},
            ]
        }
    },
    "franchise": {
        "type": "template", "intent": "franchise",
        "template": {
            "name": "mwg_franchise_info", "language": {"code": "en"},
            "components": [
                {"type": "body", "parameters": []},
                {"type": "button", "sub_type": "url", "index": "0", "parameters": [{"type": "text", "text": "franchise"}]},
            ]
        }
    },
    "freelancer": {
        "type": "template", "intent": "freelancer",
        "template": {
            "name": "mwg_freelancer_join", "language": {"code": "en"},
            "components": [
                {"type": "body", "parameters": []},
                {"type": "button", "sub_type": "url", "index": "0", "parameters": [{"type": "text", "text": "join"}]},
            ]
        }
    },
    "byob": {
        "type": "template", "intent": "byob",
        "template": {
            "name": "mwg_byob_details", "language": {"code": "en"},
            "components": [
                {"type": "body", "parameters": []},
                {"type": "button", "sub_type": "url", "index": "0", "parameters": [{"type": "text", "text": "byob"}]},
            ]
        }
    },
    "area_partner": {
        "type": "template", "intent": "area_partner",
        "template": {
            "name": "mwg_area_partner_info", "language": {"code": "en"},
            "components": [
                {"type": "body", "parameters": []},
                {"type": "button", "sub_type": "url", "index": "0", "parameters": [{"type": "text", "text": "partner"}]},
            ]
        }
    },
}

# TEMPLATE_INTENTS = set(TEMPLATES.keys())
TEMPLATE_INTENTS = {"booking"}

SYSTEM_PROMPT = """You are a helpful assistant for Mr. White Gloves (MWG), a premium doorstep car wash company in India.
- Use ONLY the provided context to answer. NEVER invent prices, services, or facts.
- Always use ₹ symbol for Indian Rupee prices.
- Keep answers concise and conversational. Use bullet points when listing items.
- Respond in the same language as the user (Hindi/English mix is fine).
- If the answer is not in the context, say: "I don't have that info. Please call us at +91 94296 91299."
"""

# Maps each intent to a knowledge category in MongoDB
INTENT_CATEGORY = {
    "byob":         "byob",
    "area_partner": "area_partner",
    "franchise":    "franchise",
    "freelancer":   "area_partner",
    "subscription": "services_subscription",
    "job_bda":      "job_bda",
    "job_fse":      "job_fse",
    "prediction":   None,
    "faq":          None,
}


def sanitize(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"(?<!\w)[\$\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_rate_limited(user_id: str) -> bool:
    now          = datetime.now(timezone.utc)
    window_start = now.timestamp() - RATE_LIMIT_WINDOW

    try:
        doc = _rate_col.find_one({"user_id": user_id})

        if not doc or doc.get("window_start", 0) < window_start:
            _rate_col.update_one(
                {"user_id": user_id},
                {"$set": {"count": 1, "window_start": now.timestamp()}},
                upsert=True
            )
            return False

        if doc["count"] >= RATE_LIMIT_MAX:
            log.warning(f"Rate limit hit for user: {user_id}")
            return True

        _rate_col.update_one({"user_id": user_id}, {"$inc": {"count": 1}})
        return False

    except Exception as e:
        log.error(f"Rate limit check failed for {user_id}: {e}")
        return False


def load_history(user_id: str) -> list:
    try:
        doc = _history_col.find_one({"user_id": user_id})
        if not doc or "messages" not in doc:
            return []

        history = []
        for msg in doc["messages"][-MAX_HISTORY:]:
            if msg["role"] == "human":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                history.append(AIMessage(content=msg["content"]))
        return history

    except Exception as e:
        log.error(f"Could not load history for {user_id}: {e}")
        return []


def save_history(user_id: str, history: list) -> None:
    try:
        messages_to_save = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                messages_to_save.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_to_save.append({"role": "ai", "content": msg.content})

        _history_col.update_one(
            {"user_id": user_id},
            {"$set": {"messages": messages_to_save, "last_updated": datetime.now(timezone.utc)}},
            upsert=True
        )
    except Exception as e:
        log.error(f"Could not save history for {user_id}: {e}")


def history_to_text(history: list) -> str:
    if not history:
        return "No previous conversation."

    lines = []
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Bot: {msg.content}")
    return "\n".join(lines)


def get_embedding(text: str) -> list:
    response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text.strip().replace("\n", " "),
    )
    return response.data[0].embedding


def get_relevant_chunks(query: str, category: str = None, top_k: int = 6) -> list:
    query_vector = get_embedding(query)

    vector_stage = {
        "$vectorSearch": {
            "index":         "knowledge_vector_index",
            "path":          "embedding",
            "queryVector":   query_vector,
            "numCandidates": max(top_k * 15, 100),
            "limit":         top_k,
        }
    }
    if category:
        vector_stage["$vectorSearch"]["filter"] = {"category": {"$eq": category}}

    pipeline = [
        vector_stage,
        {"$project": {"_id": 0, "text": 1, "category": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]

    chunks = list(_knowledge_col.aggregate(pipeline))
    log.info(f"   Vector search [{category}]: {len(chunks)} chunk(s)")

    # Not enough results — retry without category filter
    if len(chunks) < 2 and category:
        fallback_pipeline = [
            {
                "$vectorSearch": {
                    "index": "knowledge_vector_index", "path": "embedding",
                    "queryVector": query_vector, "numCandidates": max(top_k * 15, 100), "limit": top_k,
                }
            },
            {"$project": {"_id": 0, "text": 1, "category": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]
        broad = list(_knowledge_col.aggregate(fallback_pipeline))
        seen  = {c["text"] for c in chunks}
        for c in broad:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])
        chunks = chunks[:top_k]
        log.info(f"   Fallback search: {len(chunks)} chunk(s)")

    return chunks


def detect_intent_using_llm(query: str) -> str:
    prompt = f"""Classify this message into exactly one category:
booking, byob, area_partner, franchise, freelancer, subscription, prediction, faq, job_bda, job_fse

- booking      : user wants to book or schedule a car wash
- byob         : about BYOB / Be Your Own Boss kit (₹24,999 gig worker model)
- area_partner : about becoming an area/society/local subscription partner
- franchise    : about MWG franchise, ₹1.9L investment, team-based business
- freelancer   : about freelance job or part-time earning with MWG
- subscription : about monthly or yearly car wash plans/packages
- prediction   : car maintenance, parts, mechanical issues
- faq          : anything else — services, pricing, cities, general questions
- job_bda  : about Business Development Associate job at MWG office
- job_fse  : about Field Sales Executive job at MWG

Message: {query}

Reply with ONE word only. No punctuation."""

    try:
        result = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        intent = result.choices[0].message.content.strip().lower().rstrip(".,!?")
        valid  = {"booking", "byob", "area_partner", "franchise", "freelancer", "subscription", "prediction", "faq", "job_bda", "job_fse"}
        if intent in valid:
            return intent
    except Exception as e:
        log.error(f"LLM intent detection failed: {e}")

    return "faq"


def detect_intent(query: str) -> str:
    q = query.lower()

    if any(w in q for w in BYOB_WORDS):         return "byob"
    if any(w in q for w in AREA_PARTNER_WORDS):  return "area_partner"
    if any(w in q for w in FRANCHISE_WORDS):     return "franchise"
    if any(w in q for w in FREELANCER_WORDS):    return "freelancer"
    if any(w in q for w in SUBSCRIPTION_WORDS):  return "subscription"
    if any(w in q for w in BOOKING_WORDS):       return "booking"
    if any(w in q for w in JOB_BDA_WORDS):      return "job_bda"
    if any(w in q for w in JOB_FSE_WORDS):       return "job_fse"
    if any(w in q for w in PREDICT_WORDS):       return "prediction"

    return detect_intent_using_llm(query)


def call_llm_with_retry(messages: list) -> str:
    for attempt in range(1, 4):
        try:
            result = openai_client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )
            return result.choices[0].message.content
        except Exception as e:
            log.warning(f"LLM attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(2)

    return "Sorry, I'm having trouble right now. Please try again in a moment."


def get_llm_response(query: str, history: list, intent: str) -> dict:
    ROLE_ADDITIONS = {
        "prediction":   "You are also a car maintenance expert. Give short, practical advice. If unsure, recommend a professional check-up.",
        "subscription": "Focus on MWG monthly and yearly subscription plans. Highlight the value and savings clearly.",
        "faq":          "",
        "job_bda":      "You are an MWG HR assistant. Share the Business Development Associate job details clearly. At the END of every reply, always add: '📩 Interested? Please send your resume to: franchise@mrwhitegloves.com or WhatsApp: +91 94296 91299'",
        "job_fse":      "You are an MWG HR assistant. Share the Field Sales Executive job details clearly. At the END of every reply, always add: '📩 Interested? Please send your resume to: franchise@mrwhitegloves.com or WhatsApp: +91 94296 91299'",
    }

    try:
        search_query = "overview of all services and features" if any(p in query.lower() for p in BROAD_PHRASES) else query
        category     = INTENT_CATEGORY.get(intent)
        chunks       = get_relevant_chunks(search_query, category=category)

        if not chunks:
            chunks = get_relevant_chunks(search_query, category=None)

        if not chunks:
            return {"type": "fallback", "response": "I don't have information on that. Please call us at +91 94296 91299."}

        context   = "\n\n".join(c["text"] for c in chunks)
        role_note = ROLE_ADDITIONS.get(intent, "")
        system    = SYSTEM_PROMPT + (f"\n{role_note}" if role_note else "")

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Previous conversation:\n{history_to_text(history)}\n\nContext:\n{context}\n\nQuestion: {query}"}
        ]

        response = call_llm_with_retry(messages)
        return {"type": intent, "response": response}

    except Exception as e:
        log.error(f"Error in get_llm_response [{intent}]: {e}")
        return {"type": "error", "response": "Something went wrong. Please try again or call +91 94296 91299."}


def handle_message(user_id: str, question: str) -> dict:
    user_id  = sanitize(str(user_id))
    question = sanitize(question)

    if not question or len(question) < 2:
        return {"type": "validation", "response": "Please send a valid message."}

    if len(question) > MAX_MSG_LENGTH:
        return {"type": "validation", "response": f"Please keep your message under {MAX_MSG_LENGTH} characters."}

    if is_rate_limited(user_id):
        return {"type": "rate_limit", "response": "You're sending messages too fast. Please wait a minute and try again."}

    history = load_history(user_id)
    intent  = detect_intent(question)
    log.info(f"User: {user_id} | Intent: {intent} | Msg: {question[:60]}")

    if intent in TEMPLATE_INTENTS:
        return TEMPLATES[intent]

    result = get_llm_response(question, history, intent)

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=result["response"]))
    history = history[-MAX_HISTORY:]
    save_history(user_id, history)

    return result


# ==================== FASTAPI PART ====================
app = FastAPI(title="Mr. White Gloves Chatbot API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    phone: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    reply: str
    intent: str
    type: str
    options: Optional[list] = None
    template: Optional[Dict[str, Any]] = None   # Added for WhatsApp templates


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    msg = req.message.strip()
    phone = req.phone.strip()

    if not msg or len(msg) < 2:
        return ChatResponse(
            reply="Please send a valid message.",
            intent="general",
            type="validation"
        )

    if len(msg) > MAX_MSG_LENGTH:
        return ChatResponse(
            reply=f"Please keep your message under {MAX_MSG_LENGTH} characters.",
            intent="general",
            type="validation"
        )

    # Use your enhanced logic
    result = handle_message(user_id=phone, question=msg)

    # Handle Template Responses (WhatsApp)
    if result.get("type") == "template" and result.get("intent") == "booking":
        return ChatResponse(
            reply="Template response",
            intent=result.get("intent", "general"),
            type="template",
            template=result.get("template")
        )

    # Normal LLM Response
    return ChatResponse(
        reply=result.get("response", "Sorry, I couldn't process that."),
        intent=result.get("type", "faq"),
        type=result.get("type", "faq")
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MWG Enhanced RAG Chatbot"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Inhanced_code:app", host="0.0.0.0", port=8000, reload=True)