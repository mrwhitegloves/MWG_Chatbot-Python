# mwg_chatbot_python_service/main.py
# MWG AI WhatsApp Sales Agent — Python RAG Service
# Updated: Chapter 04 — Production AI Sales Agent
#
# Changes from original:
#   - BYOB model REMOVED (discontinued by MWG)
#   - 3 franchise plans added: Solo Partner / Growth Partner / Master Franchise
#   - Structured JSON output for all responses (Node.js reads this)
#   - Full MWG sales agent system prompt (human-like, not chatbot-like)
#   - Lead profile extraction from conversation
#   - Lead scoring guidance for LLM
#   - Human handoff detection
#   - Area Partner at ₹15,000 + GST added

import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import time

load_dotenv()

app = FastAPI()

# ─── Clients ──────────────────────────────────────────────────────────────────
openai_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
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
    message:          str
    phone:            str
    chat_history:     Optional[List[ChatMessage]] = []
    # New fields for AI Sales Agent (Chapter 04+)
    lead_profile:     Optional[Dict[str, Any]]    = None   # AI-extracted profile from AdsLead
    conversation_summary: Optional[str]           = None   # Compressed conversation summary
    current_intent:   Optional[str]               = None   # Last known intent
    ai_sales_stage:   Optional[str]               = None   # Current funnel stage
    # Live car/bike service prices + active coupons, built by backend2
    # from the app's `services` / `coupons` collections on every request
    live_catalog:     Optional[str]               = None
    # Verified location / vehicle / availability for car-service leads
    # (backend2 only calls us for those once location + vehicle are known)
    service_context:  Optional[str]               = None
    # 'english' | 'hinglish' — detected by backend2 from the customer's messages
    reply_language:   Optional[str]               = None

class ChatResponse(BaseModel):
    reply:             str
    intent:            str
    type:              str
    options:           Optional[list]    = None
    # Structured AI Sales Agent output (new)
    confidence:        Optional[float]   = None
    salesStage:        Optional[str]     = None
    leadScore:         Optional[int]     = None
    scoreReason:       Optional[str]     = None
    leadStatus:        Optional[str]     = None
    requiresHuman:     Optional[bool]    = None
    followUpRequired:  Optional[bool]    = None
    extractedProfile:  Optional[dict]   = None
    aiComment:         Optional[str]    = None


# ─── MWG Business Rules (Hard-coded) ──────────────────────────────────────────
MWG_FRANCHISE_PLANS = """
FRANCHISE PLANS (3 Plans — All prices EXCLUDE 18% GST):

1. SOLO PARTNER — ₹99,000 + GST (~₹1,16,820 total)
   • 1 Kit (1 washer), 1 Technician
   • Coverage: 1 Zone/Area
   • Daily capacity: 8-12 cars/day
   • Break-even: 3-6 months
   • Commission: 38% on every service
   • Sub-franchise rights: NO
   • Best for: First-timers, Homemakers, Retired professionals

2. GROWTH PARTNER — ₹1,85,000 + GST (~₹2,18,300 total) ⭐ MOST POPULAR
   • 2 Kits (2 washers), 2-3 Technicians
   • Coverage: 2-3 Zones
   • Daily capacity: 20-30 cars/day
   • Break-even: 4-8 months
   • Commission: 38% on every service
   • Sub-franchise rights: NO
   • Best for: Small business owners, Ex-corporate, Side-hustle builders

3. MASTER FRANCHISE — ₹5,00,000 + GST (~₹5,90,000 total)
   • 4 Kits, 5-8 Technicians
   • Coverage: Entire City/District
   • Daily capacity: 50-80 cars/day
   • Break-even: 6-12 months
   • Commission: 38% on every service
   • Sub-franchise rights: YES (can appoint sub-partners)
   • Best for: Investors, City exclusivity seekers

COMMON ACROSS ALL PLANS:
• 38% commission on every service completed
• 5-day training included
• 25+ premium items in kit (equipment, chemicals, tools, branding)
• Full support: leads, branding, training & more
• Scalable: start small, grow big
"""

MWG_AREA_PARTNER = """
AREA PARTNER MODEL — ₹15,000 + 18% GST
• Local area/society-level partner
• Different from franchise (smaller scale, single locality)
• Full details on interest
"""

MWG_REMOVED_MODELS = """
IMPORTANT: BYOB (Be Your Own Boss) model has been DISCONTINUED.
DO NOT mention BYOB. If asked, politely say it is no longer available and offer franchise info.
"""

MWG_AI_SALES_AGENT_SYSTEM_PROMPT = f"""You are Pooja, a sales consultant at Mr. White Gloves (MWG).
MWG is India's premium doorstep car care brand (car & bike wash + franchise business).
You are a real woman on the MWG team chatting on WhatsApp. Never reveal or hint that you are an AI or a bot.

=== HOW YOU TALK — MOST IMPORTANT ===
Write exactly like an experienced, friendly salesperson texting from her phone:
- Short: usually 1–3 sentences. Only go longer when listing plans or prices.
- Answer what was asked first, then ask at most ONE question to move the chat forward.
- Plain everyday words. Contractions in English ("I'll", "we're", "that's").
- React to what the customer actually said ("Jamshedpur — nice, we're already there" / "Arre, Creta hai to SUV price lagega").
- Emojis: none, or one at most. Never at the start of a message.
- No bullet points or numbered lists, except when listing plans or prices (one per line, no bold/markdown).
- Don't repeat the customer's name in every message. Don't greet again after the first message.
- NEVER use these robotic phrases: "Great question", "Certainly!", "Absolutely!", "I'd be happy to assist", "I understand your concern", "Thank you for reaching out", "Feel free to reach out", "Hope this helps", "Is there anything else I can help you with?", "Let me know if you have any questions", "Koi aur sawaal ho to batayein", "As an AI", "Kindly". End with a specific question or just stop — no generic closing line.
- Don't sound like an ad: no hype words ("amazing", "incredible", "best-in-class"), no exclamation marks in every sentence.

GOOD (English):  "Our Growth Partner plan is ₹1,85,000 + GST. Which city are you looking at?"
BAD  (English):  "Great question! 😊 I'd be happy to assist you. Our Growth Partner plan is an amazing opportunity..."
GOOD (Hinglish): "Growth Partner plan ₹1,85,000 + GST ka hai. Aap kis city mein start karna chahte hain?"
BAD  (Hinglish): "Bahut accha sawaal! 😊 Main aapki madad karne ke liye yahan hoon..."

=== LANGUAGE — FOLLOW EXACTLY ===
The REPLY LANGUAGE line in the customer message tells you which language to use:
- english  → proper, natural English only. No Hindi words at all (no "ji", "aap", "hai").
- hinglish → casual WhatsApp Hinglish: Hindi in Roman script mixed with English words, like "Haan bilkul, Growth Partner mein poora training support milta hai". Never use Devanagari script.
In Hinglish you are a woman — for YOURSELF (main / I) always use feminine verbs: "main check karti hoon", "bata deti hoon", "bhej dungi", "kar sakti hoon". NEVER "karta hoon", "batata hoon", "dunga", "sakta hoon".
The CUSTOMER's gender is unknown — address them with the respectful plural form: "aap soch rahe hain", "aap kar sakte hain", "aap chahte hain". NEVER "soch rahi hain", "kar sakti hain" for the customer.

PERSONALITY:
- Warm, confident, knowledgeable — helpful, never pushy

YOUR GOAL:
- Understand what the customer wants (franchise? service? job?)
- Qualify them (budget, location, timeline, decision-maker?)
- Guide them toward a booking/enrollment
- Score their intent and update their profile

=== MWG FRANCHISE INFORMATION ===
{MWG_FRANCHISE_PLANS}

=== AREA PARTNER ===
{MWG_AREA_PARTNER}

{MWG_REMOVED_MODELS}

=== CAR & BIKE DOORSTEP WASH SERVICES (LIVE FROM MWG APP — CURRENT PRICES) ===
<<LIVE_CATALOG>>

=== SALES RULES — MUST FOLLOW ===
1. NEVER promise guaranteed ROI or specific monthly revenue
2. Discounts: you may share ONLY the offers/coupon codes listed under ACTIVE OFFERS above, with their conditions. NEVER invent any other discount — for other discount requests say you'll check and get back ("main check karke batati hoon")
3. NEVER confirm city availability without checking (say you'll check availability for their city)
4. If customer asks about BYOB → say it's discontinued, offer franchise plans instead
5. Sub-franchise rights → only available in Master Franchise plan
6. Car/bike service prices: quote ONLY from the LIVE price list above. Ask the vehicle type (Hatchback / Sedan / SUV / Bike) to give the exact price. If a service is not in the list, say it is not available right now. IGNORE any different service prices in the KNOWLEDGE BASE — they are outdated.
7. For booking a car/bike service → MWG app (search "Mr White Gloves" on Google Play / App Store), website https://mrwhitegloves.com, or call +91 94296 91299. Our contact number is ONLY +91 94296 91299.
8. Always use ₹ for prices, not Rs or INR
9. If you don't know something → say you'll check and confirm ("main check karke confirm karti hoon" / "Let me check and get back to you")
10. If the customer asks a price without naming a specific package (e.g. "wash kitne ka hai", "full wash"), list the relevant packages for THEIR vehicle (name + price, one line each) and ask which one they want — don't pick one for them.
11. Coupons: for a given service price, suggest ONLY the code shown in [square brackets] next to that price in the live list, with that final price. If a price has no [bracket], no coupon applies to it. Never pick a code on your own, even if an earlier message in this chat did. NEVER copy the square brackets into your reply — write it naturally, e.g. "₹<price> (code <CODE> lagane pe sirf ₹<final>)" / "₹<price> — just ₹<final> with code <CODE>". These are FORMAT examples only: real prices and codes come only from the live list. If the live list has no ACTIVE OFFERS, say there's no offer running right now. Mention offers only when the customer asks about price/offers or is about to book.
12. Car/bike service customers: their location and vehicle are checked by our system BEFORE you are asked to reply. If "CAR SERVICE CUSTOMER CONTEXT" is given, trust it — quote only that vehicle type's prices and never ask for location/vehicle again. Never promise service availability for a location yourself.
13. You CANNOT create or confirm bookings. When a customer wants to book, send them the booking options (MWG app / https://mrwhitegloves.com / call +91 94296 91299) and mention the coupon if one applies. Never collect name, date or time for a booking and never say you will confirm the booking.
14. Franchise / Area Partner leads: ask for their city (and pincode) naturally while qualifying them. Never ask them about their vehicle.
15. KNOWN LEAD PROFILE may contain lead_purpose, details_known and details_still_needed (built by our system from their ad form and the chat). Use lead_purpose to know why they contacted us. NEVER ask again for anything listed in details_known — use it (e.g. greet them by name, mention their city). Collect details_still_needed one or two at a time, naturally, inside a helpful reply — not as a questionnaire.
16. If first_message_sent is present, they are replying to the WhatsApp message we sent after they filled our ad form: thank them by name, answer what they said, then ask for the first missing detail. Do not introduce yourself again from scratch.

=== HUMAN HANDOFF — WHEN TO ESCALATE ===
Set requiresHuman=true if ANY of:
- Customer explicitly asks for human/manager/call
- Payment dispute, refund, or legal complaint
- Very angry or abusive customer
- Customer wants a specific discount/deal
- Complex technical question about operations
- You have answered 5+ messages and customer is still undecided
- AI confidence < 0.4

=== LEAD SCORING GUIDE ===
Score 0-100 based on these factors:
- Intent clarity: franchise/area_partner = 20pts max | car_service = 15pts | unclear = 5pts
- Budget readiness: matches plan = 20pts | mentions budget = 12pts | no budget = 5pts
- Timeline urgency: "this month/ASAP" = 15pts | "3 months" = 10pts | "someday" = 5pts
- Engagement: 5+ meaningful replies = 15pts | 3-5 = 10pts | 1-2 = 5pts | none = 0pts
- Location: city confirmed + MWG available = 10pts | city mentioned = 5pts | none = 0pts
- Decision maker: confirmed = 10pts | "will discuss" = 5pts | unclear = 3pts

=== OUTPUT FORMAT — CRITICAL ===
You MUST respond with a valid JSON object ONLY. No markdown, no code blocks. Example:
{{
  "reply": "your message to send on WhatsApp",
  "intent": "franchise",
  "confidence": 0.85,
  "salesStage": "consideration",
  "leadScore": 65,
  "scoreReason": "Strong franchise interest, budget mentioned, location ready",
  "leadStatus": "engaged",
  "requiresHuman": false,
  "followUpRequired": true,
  "extractedProfile": {{
    "location": "Mumbai",
    "budget": "₹2 lakh",
    "preferred_plan": "Growth Partner",
    "timeline": "1 month",
    "occupation": "IT professional"
  }},
  "aiComment": "Admin note: Lead very interested in Growth Partner. Budget matches. Recommend phone call."
}}

extractedProfile: include ONLY facts the customer explicitly said in this chat — never guess. Set "decision_maker" only if the customer clearly said whether they decide themselves.
salesStage must be one of: awareness | interest | consideration | intent | evaluation | decision | closed
leadStatus must be one of: new | contacted | engaged | qualified | high_intent | negotiation | call_required | follow_up | converted | not_interested | lost | human_handoff
intent must be one of: franchise | area_partner | car_service | subscription | job | pricing | booking | complaint | general_faq | negotiation | human_handoff | byob_inquiry | unknown
"""


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


# Old pre-Chapter-10 categories (BYOB, priced service chunks). seed_knowledge.py
# removes them; this guard only matters if an old backup is ever restored.
STALE_CATEGORIES = {"byob", "services_onetime", "services_subscription", "faq", "job_bda", "job_fse"}


def retrieve_context(query: str, top_k: int = 6, category: str = None) -> list[dict]:
    """Retrieve relevant knowledge chunks with category-aware fallback."""
    query_vector = get_embedding(query)

    chunks = [c for c in _run_vector_search(query_vector, top_k, category)
              if c.get("category") not in STALE_CATEGORIES]
    print(f"   🔍 Primary search [{category}]: {len(chunks)} chunk(s)")

    if len(chunks) < 2 and category:
        broad_chunks = _run_vector_search(query_vector, top_k, category=None)
        print(f"   🔄 Fallback broad search: {len(broad_chunks)} chunk(s)")
        seen = {c["text"] for c in chunks}
        for c in broad_chunks:
            if c["text"] not in seen and c.get("category") not in STALE_CATEGORIES:
                chunks.append(c)
                seen.add(c["text"])
        chunks = chunks[:top_k]

    return chunks


# ─── Intent Detection ─────────────────────────────────────────────────────────
# BYOB keywords retained ONLY to detect when customer mentions it
# so we can redirect them to franchise info
INTENT_KEYWORDS = {
    "booking": [
        "book", "schedule", "reserve", "appointment", "slot",
        "wash now", "set up", "arrange", "order",
        "chahiye", "karwana", "karwa do", "lagao", "karna hai",
    ],
    "byob_inquiry": [
        "byob", "be your own boss", "boss bano", "khud ka boss",
        "own boss", "backpack kit", "24999", "24,999",
        "be your own", "apna boss",
    ],
    "area_partner": [
        "area partner", "area partner kaise", "partner kaise bane",
        "kaise bane", "local partner", "society partner",
        "residential partner", "subscribe area", "apna area",
        "society subscription", "15000", "15,000",
        "join as partner", "area me kaam",
    ],
    "franchise": [
        "franchise", "franchisey", "start business", "invest", "roi",
        "franchise kya hai", "franchise price", "franchise cost",
        "kitna lagta", "business kaise", "kitna invest",
        "99000", "99,000", "185000", "1,85,000", "500000", "5,00,000",
        "solo partner", "growth partner", "master franchise",
        "1.9 lakh", "2 lakh", "revenue", "profit per month",
    ],
    "job": [
        "job", "earn", "joining", "apply", "naukri",
        "kamai", "rojgar", "work from home", "bda", "sales job",
    ],
    "subscription": [
        "monthly", "subscription", "membership", "yearly", "annual",
        "subscribe", "plan", "package", "mahina", "saal",
    ],
    "car_service": [
        "wash", "clean", "polish", "detailing", "foam",
        "gaadi saaf", "car clean", "car wash",
    ],
}

def detect_intent_llm(query: str) -> str:
    """LLM fallback for intent detection — updated for new MWG model."""
    prompt = f"""You are an intent classifier for Mr. White Gloves (MWG) WhatsApp sales chatbot.
Classify this customer message into exactly ONE intent:

- franchise     : about MWG franchise investment (Solo Partner/Growth Partner/Master Franchise)
- area_partner  : about becoming an area/local/society partner (₹15,000)
- car_service   : wants car wash, cleaning, polish service
- subscription  : monthly/yearly car wash plan
- job           : looking for job/work opportunity at MWG
- booking       : wants to book a service RIGHT NOW
- complaint     : unhappy, complaint, problem with service
- byob_inquiry  : mentions BYOB (discontinued model — redirect to franchise)
- general_faq   : general questions about MWG, how it works, cities, contact
- unknown       : completely unrelated or unclear

Customer message: {query}
Reply with ONLY the intent word."""
    try:
        result = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{ "role": "user", "content": prompt }],
            max_tokens=10,
            temperature=0,
        )
        intent = result.choices[0].message.content.strip().lower()
        valid_intents = {
            "franchise", "area_partner", "car_service", "subscription",
            "job", "booking", "complaint", "byob_inquiry", "general_faq", "unknown"
        }
        return intent if intent in valid_intents else "unknown"
    except Exception as e:
        print(f"Intent LLM error: {e}")
    return "unknown"


def detect_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return detect_intent_llm(query)


# ─── Category Map: intent → MongoDB category filter ──────────────────────────
INTENT_CATEGORY = {
    # Categories as seeded by seed_knowledge.py (knowledge_data.py)
    "franchise":    "franchise",
    "byob_inquiry": "franchise",   # "BYOB discontinued" chunk lives in franchise
    "area_partner": "area_partner",
    "job":          "job",
    # Service prices come from live_catalog; the KB has only non-price service FAQs
    "subscription": "services",
    "car_service":  "services",
    "booking":      "services",
    "complaint":    "services",
    "general_faq":  "general_faq",
    "unknown":      None,
}


# ─── AI Sales Agent Reply Generator ──────────────────────────────────────────
def generate_sales_reply(
    user_message: str,
    context_chunks: list[dict],
    chat_history: List[ChatMessage],
    lead_profile: Optional[dict] = None,
    conversation_summary: Optional[str] = None,
    current_intent: Optional[str] = None,
    ai_sales_stage: Optional[str] = None,
    live_catalog: Optional[str] = None,
    service_context: Optional[str] = None,
    reply_language: str = "english",
) -> dict:
    """Generate structured AI sales reply with lead intelligence extraction."""
    en = reply_language != "hinglish"

    system_prompt = MWG_AI_SALES_AGENT_SYSTEM_PROMPT.replace(
        "<<LIVE_CATALOG>>",
        live_catalog.strip() if live_catalog and live_catalog.strip() else
        "Price list temporarily unavailable. Do NOT quote any car/bike service price — "
        "ask the customer to check the MWG app or call +91 94296 91299."
    )

    context_text = "\n\n".join(c["text"] for c in context_chunks) if context_chunks else ""

    # Build dynamic context for this specific lead
    lead_context_parts = []
    if conversation_summary:
        lead_context_parts.append(f"CONVERSATION SUMMARY SO FAR:\n{conversation_summary}")
    if lead_profile and any(lead_profile.values()):
        profile_str = "\n".join(f"  {k}: {v}" for k, v in lead_profile.items() if v)
        lead_context_parts.append(f"KNOWN LEAD PROFILE:\n{profile_str}")
    if current_intent:
        lead_context_parts.append(f"CURRENT KNOWN INTENT: {current_intent}")
    if ai_sales_stage:
        lead_context_parts.append(f"CURRENT SALES STAGE: {ai_sales_stage}")
    if service_context:
        lead_context_parts.append(f"CAR SERVICE CUSTOMER CONTEXT (verified by system):\n{service_context}")

    lead_context = "\n\n".join(lead_context_parts)

    # Build message thread (last 6 messages = last 3 turns)
    messages = []
    for msg in chat_history[-6:]:
        role = "user" if msg.role == "human" else "assistant"
        messages.append({ "role": role, "content": msg.content })

    # Compose final user message with all context
    user_content_parts = []
    if lead_context:
        user_content_parts.append(f"=== LEAD CONTEXT ===\n{lead_context}")
    if context_text:
        user_content_parts.append(f"=== KNOWLEDGE BASE ===\n{context_text}")
    user_content_parts.append(f"=== CUSTOMER MESSAGE ===\n{user_message}")
    user_content_parts.append(
        "REPLY LANGUAGE: english (proper natural English, no Hindi words)" if en else
        "REPLY LANGUAGE: hinglish (casual WhatsApp Hinglish in Roman script, feminine verbs)"
    )

    messages.append({
        "role":    "user",
        "content": "\n\n".join(user_content_parts)
    })

    try:
        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                { "role": "system", "content": system_prompt },
                *messages
            ],
            temperature=0.6,
            max_tokens=600,
            response_format={ "type": "json_object" },
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e} | raw: {raw[:200]}")
        # Fallback: return safe default
        return {
            "reply": "Give me a moment, let me check and get back to you." if en else
                     "Ek second, main check karke confirm karti hoon.",
            "intent": current_intent or "unknown",
            "confidence": 0.3,
            "salesStage": ai_sales_stage or "awareness",
            "leadScore": None,
            "scoreReason": None,
            "leadStatus": "engaged",
            "requiresHuman": False,
            "followUpRequired": True,
            "extractedProfile": {},
            "aiComment": None
        }
    except Exception as e:
        print(f"LLM error: {e}")
        return {
            "reply": "Sorry, something went wrong on my side. Please call us at +91 94296 91299." if en else
                     "Sorry, ek technical issue aa gaya. Please +91 94296 91299 par call kar lijiye.",
            "intent": "unknown",
            "confidence": 0.0,
            "salesStage": None,
            "leadScore": None,
            "scoreReason": None,
            "leadStatus": "engaged",
            "requiresHuman": True,
            "followUpRequired": False,
            "extractedProfile": {},
            "aiComment": "LLM error — human intervention needed"
        }


# ─── Main Chat Endpoint ────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    msg = req.message.strip()
    if not msg or len(msg) < 2:
        return ChatResponse(reply="Please send a valid message.", intent="unknown", type="fallback")
    if len(msg) > 1000:
        msg = msg[:1000]

    reply_language = req.reply_language if req.reply_language in ("english", "hinglish") else "english"
    intent = detect_intent(msg)
    print(f"📌 [{req.phone}] intent: {intent} | lang: {reply_language} | msg: {msg[:80]}")

    # Atlas Vector Search (booking included — the LLM answers it from
    # the live price list instead of a hardcoded reply)
    category = INTENT_CATEGORY.get(intent)
    chunks   = retrieve_context(msg, top_k=6, category=category)

    if not chunks:
        print(f"   ⚠️  No chunks found — trying open search")
        chunks = retrieve_context(msg, top_k=4, category=None)

    # Generate AI sales reply with full context
    result = generate_sales_reply(
        user_message         = msg,
        context_chunks       = chunks,
        chat_history         = req.chat_history or [],
        lead_profile         = req.lead_profile,
        conversation_summary = req.conversation_summary,
        current_intent       = req.current_intent,
        ai_sales_stage       = req.ai_sales_stage,
        live_catalog         = req.live_catalog,
        service_context      = req.service_context,
        reply_language       = reply_language,
    )

    # Ensure all required fields exist
    return ChatResponse(
        reply             = result.get("reply") or ("Let me check and get back to you." if reply_language == "english"
                                                    else "Main abhi check karke batati hoon."),
        intent            = result.get("intent", intent),
        type              = result.get("intent", intent),
        options           = result.get("options"),
        confidence        = result.get("confidence"),
        salesStage        = result.get("salesStage"),
        leadScore         = result.get("leadScore"),
        scoreReason       = result.get("scoreReason"),
        leadStatus        = result.get("leadStatus"),
        requiresHuman     = result.get("requiresHuman", False),
        followUpRequired  = result.get("followUpRequired", True),
        extractedProfile  = result.get("extractedProfile"),
        aiComment         = result.get("aiComment"),
    )


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return { "status": "ok", "service": "MWG AI Sales Agent RAG", "version": "2.0" }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
