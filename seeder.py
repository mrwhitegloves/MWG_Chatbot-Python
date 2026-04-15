# python-service/seeder.py
# Run once: python seeder.py
# Re-run whenever you update any knowledge JSON file

import os, json, time
from pathlib import Path
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client     = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
mongo      = MongoClient(os.getenv("MONGODB_URI"))
collection = mongo[os.getenv("MONGODB_DB", "mwg_chatbot")]["knowledge_chunks"]

KNOWLEDGE_DIR = Path("knowledge")
FILES = [
    "services_onetime.json",
    "services_subscription.json",
    "franchise_modal.json",
    "byob_modal.json",
    "area_partner_modal.json",
    "faqs.json",
]

def get_embeddings_batch(texts: list[str]) -> list:
    cleaned  = [t.strip().replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=cleaned,
    )
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

def seed():
    print("🌱 Starting knowledge base seeder...\n")

    # Clear old data
    deleted = collection.delete_many({})
    print(f"🗑️  Cleared {deleted.deleted_count} old chunks\n")

    all_records = []

    for filename in FILES:
        filepath = KNOWLEDGE_DIR / filename
        if not filepath.exists():
            print(f"⚠️  Skipping missing: {filename}")
            continue

        entries = json.loads(filepath.read_text(encoding="utf-8"))
        for entry in entries:
            # Q+A combined = better semantic search
            text = f"Q: {entry['question']}\nA: {entry['answer']}"
            all_records.append({
                "id":       entry["id"],
                "text":     text,
                "category": entry["category"],
                "source":   filename,
            })
        print(f"✅ Loaded {len(entries)} entries from {filename}")

    print(f"\n⚙️  Generating embeddings for {len(all_records)} chunks...")

    # Batch embed (100 per batch)
    BATCH = 100
    for i in range(0, len(all_records), BATCH):
        batch      = all_records[i:i+BATCH]
        texts      = [r["text"] for r in batch]
        embeddings = get_embeddings_batch(texts)

        for record, emb in zip(batch, embeddings):
            record["embedding"] = emb

        print(f"   Batch {i//BATCH + 1} done ({min(i+BATCH, len(all_records))}/{len(all_records)})")
        time.sleep(0.3)  # avoid rate limits

    collection.insert_many(all_records)
    print(f"\n✅ Inserted {len(all_records)} chunks into MongoDB Atlas")
    print("\n📋 NEXT: Create Atlas Vector Search index (see ATLAS_SETUP.md)")
    mongo.close()

if __name__ == "__main__":
    seed()