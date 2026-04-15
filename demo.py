from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document 
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

with open("data.txt", "r", encoding="utf-8") as f:
    data_text = f.read()

with open ("franchise.txt",'r',encoding="utf-8")as f:
    franchise_text=f.read()
    
docs = [Document(page_content=data_text),
        Document(page_content=franchise_text)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False # Recommended for OpenRouter/non-OpenAI providers
)

FAISS_INDEX_DIR = "faiss_index"
SOURCE_FILES = ["data.txt", "franchise.txt"]

def index_is_outdated():
    """Return True if any source file is newer than the saved index."""
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
 
    # No index saved yet → must build
    if not os.path.exists(index_file):
        return True
 
    index_time = os.path.getmtime(index_file)
 
    # If any source file was edited after the index was built → rebuild
    for src in SOURCE_FILES:
        if os.path.exists(src) and os.path.getmtime(src) > index_time:
            return True
 
    return False  # nothing changed → safe to load
 
if index_is_outdated():
    print("Building index from data.txt + franchise.txt ...")
    vector_store = FAISS.from_documents(chunks, embedding)
    vector_store.save_local(FAISS_INDEX_DIR)
    print("Index built and saved.")
else:
    print("Loading saved index ...")
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )
    print("Ready!")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8}
)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
)

BOOKING_WORDS = ["book", "schedule", "reserve", "appointment", "slot",
                 "wash now", "set up", "arrange", "order"]

#Future words that  we will be using for prediction
PREDICT_WORDS = ["brake", "tyre", "tire", "battery", "oil", "filter",
                 "replace", "fail", "problem", "noise", "health",
                 "worn", "check", "spark", "when should", "predict"]

FRANCHISE_WORDS = ["franchise", "business", "partner", "start business"]
FREELANCER_WORDS = ["freelancer", "job", "work", "earn", "join"]

def detect_intent_llm(query):
    prompt = f"""
Classify the user query into one of these:
booking, faq, prediction, general

Query: {query}

Answer ONLY one word.
"""
    try:
        intent = llm.invoke(prompt).content.strip().lower()
        if intent in ["booking", "faq", "prediction", "general"]:
            return intent
    except Exception as e:
        print(f"Intent detection failed: {e}")
    return "general"

def detect_intent(query):
    q = query.lower()
    if any(w in q for w in FRANCHISE_WORDS):
        return "franchise"
    elif any(w in q for w in FREELANCER_WORDS):
        return "freelancer"
    elif any(w in q for w in BOOKING_WORDS):
        return "booking"
    if any(w in q for w in PREDICT_WORDS):
        return "prediction"
    return detect_intent_llm(query)

def is_broad_query(query):
    keywords = ["all", "everything", "features", "provide", "services"]
    return any(k in query.lower() for k in keywords)

def booking_response():
    return {
        "type": "booking",
        "response": "Choose a service to book:",
        "options": [
            {"name": "Basic Wash",    "link": "https://mrwhitegloves.com/basic"},
            {"name": "Premium Wash",  "link": "https://mrwhitegloves.com/premium"}
        ]
    }

#Future Plan 
def prediction_response(query, chat_history):
    try:
        retrieved_docs = retriever.invoke(query)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        prompt = f"""
You are a car maintenance expert.
Answer briefly and practically.

Context:
{context_text}

Question:
{query}
"""
        response = llm.invoke(prompt).content
        return {
            "type": "prediction",
            "response": response
        }

    except Exception as e:
        return {
            "type": "error",
            "response": f"Sorry, I couldn't process your car part query right now. Please call us at +91-98765-43210. (Error: {e})"
        }

def faq_response(query, chat_history):
    try:
        if is_broad_query(query):
            context_text = "\n\n".join(doc.page_content for doc in chunks)
        else:
            retrieved_docs = retriever.invoke(query)

            if not retrieved_docs:
                return {
                    "type": "fallback",
                    "response": "I don't know based on available data."
                }

            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        prompt = f"""
You are a helpful assistant for Mr White Gloves.

If the user asks for:
- all features
- everything you provide
- complete details

Then include ALL categories:
- services
- plans
- franchise
- freelancer

If not found, say "I don't know".

Context:
{context_text}

Question:
{query}
"""

        response = llm.invoke(prompt).content

        return {
            "type": "faq",
            "response": response
        }

    except Exception as e:
        return {
            "type": "error",
            "response": f"Sorry, something went wrong. Please try again. (Error: {e})"
        }

chat_history = []

try:
    with open("chat_history.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Human:"):
                chat_history.append(HumanMessage(content=line[6:].strip()))
            elif line.startswith("AI:"):
                chat_history.append(AIMessage(content=line[3:].strip()))
except FileNotFoundError:
    print("No previous chat history found. Starting fresh.")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    # FIX: Stronger input validation
    if not question.strip() or len(question.strip()) < 2:
        print("Bot: Please ask a valid question.\n")
        continue

    if len(question) > 150:
        print("Bot: Please keep your question under 150 characters.\n")
        continue

    intent = detect_intent(question)

   
    if intent == "booking":
        result = booking_response()
    elif intent == "prediction":
        result = prediction_response(question, chat_history)
    elif intent in ["franchise", "freelancer"]:
        result = faq_response(question, chat_history)
    else:
        result = faq_response(question, chat_history)

    print("\nBot:", result["response"])
    if result["type"] == "booking" and "options" in result:
        for opt in result["options"]:
            print(f"  • {opt['name']}: {opt['link']}")
    print()


    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result["response"]))
    chat_history = chat_history[-10:]

    try:
        with open("chat_history.txt", "a", encoding="utf-8") as f:
            f.write(f"Human: {question}\n")
            f.write(f"AI: {result['response']}\n")
    except Exception as e:
        print(f"Warning: Could not save chat history. ({e})")