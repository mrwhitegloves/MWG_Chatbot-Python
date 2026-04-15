# python-service/main.py
# Enhanced demo.py → Production FastAPI RAG Service
# Uses MongoDB Atlas Vector Search instead of FAISS + .txt files

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import time

load_dotenv()

app    = FastAPI()

# ─── Clients ──────────────────────────────────────────────────────────────────
openai_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # keeping your OpenRouter setup
)

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db           = mongo_client[os.getenv("MONGODB_DB", "mwg_chatbot")]
collection   = db["knowledge_chunks"]

print("✅ Connected to MongoDB Atlas")


# ─── Schemas ──────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role:    str      # "human" | "ai"
    content: str

class ChatRequest(BaseModel):
    message:      str
    phone:        str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    reply:   str
    intent:  str
    type:    str
    options: Optional[list] = None


# ─── Embedding (OpenRouter) ───────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text.strip().replace("\n", " "),
    )
    return response.data[0].embedding


# ─── Atlas Vector Search ──────────────────────────────────────────────────────
def _run_vector_search(query_vector: list, top_k: int, category: str = None) -> list[dict]:
    """Execute a single Atlas vector search pipeline."""
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
        vector_stage["$vectorSearch"]["filter"] = {
            "category": { "$eq": category }
        }
    pipeline = [
        vector_stage,
        { "$project": { "_id": 0, "text": 1, "category": 1,
                        "score": { "$meta": "vectorSearchScore" } } }
    ]
    return list(collection.aggregate(pipeline))


def retrieve_context(query: str, top_k: int = 6, category: str = None) -> list[dict]:
    """Retrieve relevant knowledge chunks with category-aware fallback.

    Strategy:
        1. Search with the specific category filter (precise).
        2. If that returns < 2 results, retry WITHOUT the filter (broad recall).
        3. De-duplicate by text and return top_k results.
    """
    query_vector = get_embedding(query)

    # Primary search (category-filtered if available)
    chunks = _run_vector_search(query_vector, top_k, category)
    print(f"   🔍 Primary search [{category}]: {len(chunks)} chunk(s)")

    # Fallback: broaden to all categories when primary returns too few results
    if len(chunks) < 2 and category:
        broad_chunks = _run_vector_search(query_vector, top_k, category=None)
        print(f"   🔄 Fallback broad search: {len(broad_chunks)} chunk(s)")
        # Merge: primary hits first, then unique broad hits
        seen = {c["text"] for c in chunks}
        for c in broad_chunks:
            if c["text"] not in seen:
                chunks.append(c)
                seen.add(c["text"])
        chunks = chunks[:top_k]

    return chunks


# ─── Intent Detection (your existing logic, enhanced) ─────────────────────────
INTENT_KEYWORDS = {
    "booking": [
        "book", "schedule", "reserve", "appointment", "slot",
        "wash now", "set up", "arrange", "order",
        # Hindi / transliterated
        "chahiye", "karwana", "karwa do", "lagao", "karna hai",
    ],
    "byob": [
        "byob", "be your own boss", "boss bano", "khud ka boss",
        "own boss", "backpack kit", "gig worker", "24999", "24,999",
        "be your own", "apna boss",
    ],
    "area_partner": [
        "area partner", "area partner kaise", "partner kaise bane",
        "kaise bane", "local partner", "society partner",
        "residential partner", "subscribe area", "apna area",
        "society subscription", "3000", "5000", "3,000", "5,000",
        "join as partner", "area me kaam",
    ],
    "franchise": [
        "franchise", "franchisey", "start business", "invest", "roi",
        "franchise kya hai", "franchise price", "franchise cost",
        "kitna lagta", "business kaise", "kitna invest", "1,90,000",
        "190000", "1.9 lakh", "2 lakh", "revenue", "profit per month",
    ],
    "freelancer": [
        "freelancer", "job", "earn", "joining", "apply",
        "kamai", "rojgar", "work from home",
    ],
    "subscription": [
        "monthly", "subscription", "membership", "yearly", "annual",
        "subscribe", "plan", "package", "mahina", "saal", "9999", "1299",
    ],
    "prediction": [
        "brake", "tyre", "tire", "battery", "oil", "filter",
        "replace", "fail", "problem", "noise", "worn",
        "check", "spark", "when should", "predict",
    ],
}

def detect_intent_llm(query: str) -> str:
    """LLM fallback for intent detection — knows all valid MWG intent categories"""
    prompt = f"""You are an intent classifier for a Mr. White Gloves (doorstep car wash) WhatsApp chatbot.
Classify the following customer query into exactly ONE of these intents:
- booking       : customer wants to book a car wash service
- byob          : about BYOB (Be Your Own Boss) model, gig worker kit, 24999 investment
- area_partner  : about becoming an area partner, local subscription business, society-level partner
- franchise     : about MWG franchise, 1.9L investment, starting a team-based business
- freelancer    : about freelance work, job, earning opportunity (not franchise/byob/area_partner)
- subscription  : about monthly/yearly car wash plans and packages
- faq           : general questions about services, safety, eco-friendly, cities available, contact
- prediction    : car maintenance prediction, part replacement, mechanical queries
- general       : anything else not covered above

Query: {query}
Answer ONLY one word (the intent name)."""
    try:
        result = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{ "role": "user", "content": prompt }],
            max_tokens=10,
            temperature=0,
        )
        intent = result.choices[0].message.content.strip().lower()
        valid_intents = {
            "booking", "byob", "area_partner", "franchise",
            "freelancer", "subscription", "faq", "prediction", "general"
        }
        if intent in valid_intents:
            return intent
    except Exception as e:
        print(f"Intent LLM error: {e}")
    return "general"

def detect_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return detect_intent_llm(query)


# ─── Category Map: intent → MongoDB category filter ───────────────────────────
# None = search across ALL categories (broader recall)
INTENT_CATEGORY = {
    "byob":         "byob",
    "area_partner": "area_partner",
    "franchise":    "franchise",
    "freelancer":   "area_partner",
    "subscription": "services_subscription",
    "prediction":   None,
    "faq":          None,
    "general":      None,
}


# ─── LLM Reply Generator ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful and knowledgeable WhatsApp assistant for Mr. White Gloves (MWG), a premium doorstep car wash company in India.

Your job:
- Answer customer questions about MWG services, pricing, BYOB model, Area Partner model, and Franchise model.
- Use the provided CONTEXT to answer. The context contains the most relevant knowledge.
- If the context covers the question, give a clear, complete, friendly answer.
- You may combine multiple context snippets to give a full answer.
- If the context is partially relevant, use what is available and answer as best you can.
- Keep replies SHORT and conversational — this is WhatsApp. Use bullet points when listing items.
- Always use ₹ symbol for all Indian Rupee prices.
- Respond in the same language as the user (Hindi/English mix is fine).
- Only if the context has ZERO relevant information about the topic, say: "I don't have that info. Please call +91 70048 10369."
- NEVER invent prices, features, or facts not present in the context.
"""

def generate_reply(
    user_message: str,
    context_chunks: list[dict],
    chat_history: List[ChatMessage]
) -> str:
    context_text = "\n\n".join(c["text"] for c in context_chunks)

    # Build messages: history + new message with context
    messages = []
    for msg in chat_history[-6:]:  # last 3 turns
        role = "user" if msg.role == "human" else "assistant"
        messages.append({ "role": role, "content": msg.content })

    messages.append({
        "role":    "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"
    })

    response = openai_client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            *messages
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


# ─── Booking Response ──────────────────────────────────────────────────────────
def booking_response() -> dict:
    return {
        "type":    "booking",
        "intent":  "booking",
        "reply":   "Choose a service to book:",
        "options": [
            { "name": "Foam Wash & Vacuum Clean",  "price": "₹299",  "link": "https://mrwhitegloves.com/basic" },
            { "name": "Wash + Int/Ext Polish",     "price": "₹699",  "link": "https://mrwhitegloves.com/polish" },
            { "name": "Deep Cleaning & Detailing", "price": "₹2499", "link": "https://mrwhitegloves.com/detailing" },
            { "name": "Monthly Subscription",      "price": "₹1299", "link": "https://mrwhitegloves.com/monthly" },
            { "name": "Yearly Package",            "price": "₹9999", "link": "https://mrwhitegloves.com/yearly" },
        ]
    }


# ─── Main Chat Endpoint ────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    # Input validation (keeping your existing checks)
    msg = req.message.strip()
    if not msg or len(msg) < 2:
        return ChatResponse(reply="Please send a valid message.",
                            intent="general", type="fallback")
    if len(msg) > 500:
        return ChatResponse(reply="Please keep your message under 500 characters.",
                            intent="general", type="fallback")

    intent = detect_intent(msg)
    print(f"📌 [{req.phone}] intent: {intent} | msg: {msg}")

    # Booking → no RAG needed
    if intent == "booking":
        result = booking_response()
        return ChatResponse(**result)

    # All others → Atlas Vector Search
    category = INTENT_CATEGORY.get(intent)  # None = search all categories
    chunks   = retrieve_context(msg, top_k=6, category=category)

    # Last-resort: if still no chunks found, try a completely open search
    if not chunks:
        print(f"   ⚠️  No chunks found — trying open search without filters")
        chunks = retrieve_context(msg, top_k=6, category=None)

    if not chunks:
        return ChatResponse(
            reply="I don't have that info. Please call +91 70048 10369.",
            intent=intent, type="fallback"
        )

    reply = generate_reply(msg, chunks, req.chat_history)
    return ChatResponse(reply=reply, intent=intent, type=intent, options=None)


# ─── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return { "status": "ok", "service": "MWG Python RAG" }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)