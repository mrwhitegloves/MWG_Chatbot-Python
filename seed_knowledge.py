# mwg_chatbot_python_service/seed_knowledge.py
# MWG AI WhatsApp Sales Agent — Knowledge Base seeder (Chapter 10)
#
# Replaces the `knowledge_chunks` collection with the chunks in knowledge_data.py:
#   1. Backs up the current collection (with embeddings) to
#      knowledge_chunks_backup_<YYYYMMDD_HHMMSS>
#   2. Embeds every chunk (same model main.py uses for search)
#   3. Upserts chunks by `id`
#   4. Deletes chunks whose `id` is no longer in knowledge_data.py
# The Atlas vector index (knowledge_vector_index) stays on the collection.
#
# Usage:
#   py -3.10 seed_knowledge.py            → apply
#   py -3.10 seed_knowledge.py --dry-run  → only show what would change
#   py -3.10 seed_knowledge.py --no-backup

import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne

from knowledge_data import KNOWLEDGE

load_dotenv()

EMBED_MODEL   = "openai/text-embedding-3-small"   # must match main.py get_embedding()
EMBED_DIMS    = 1536                              # must match knowledge_vector_index
BATCH_SIZE    = 20
VALID_CATS    = {"franchise", "area_partner", "services", "job", "general_faq"}
SOURCE_TAG    = "knowledge_data.py"


def validate(chunks):
    ids = [c["id"] for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        sys.exit(f"❌ Duplicate ids in knowledge_data.py: {sorted(dupes)}")
    for c in chunks:
        if c["category"] not in VALID_CATS:
            sys.exit(f"❌ {c['id']}: unknown category '{c['category']}' (allowed: {sorted(VALID_CATS)})")
        if not c["text"].strip():
            sys.exit(f"❌ {c['id']}: empty text")
        if "byob" in c["text"].lower() and c["id"] != "fr_018":
            sys.exit(f"❌ {c['id']}: mentions BYOB — only the discontinued-redirect chunk may")
        if "70048" in c["text"]:
            sys.exit(f"❌ {c['id']}: contains the old phone number")


def embed_all(client, texts):
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t.strip().replace("\n", " ") for t in texts[i:i + BATCH_SIZE]]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                vectors.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"   ⚠️  Embedding batch failed ({e}) — retrying...")
                time.sleep(3)
        print(f"   🧠 Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    bad = [i for i, v in enumerate(vectors) if len(v) != EMBED_DIMS]
    if bad:
        sys.exit(f"❌ Unexpected embedding size for chunks {bad} (expected {EMBED_DIMS})")
    return vectors


def main():
    dry_run   = "--dry-run" in sys.argv
    no_backup = "--no-backup" in sys.argv

    validate(KNOWLEDGE)

    mongo = MongoClient(os.getenv("MONGODB_URI"))
    db    = mongo[os.getenv("MONGODB_DB", "mwg_chatbot")]
    col   = db["knowledge_chunks"]

    existing     = {d["id"]: d for d in col.find({}, {"embedding": 0}) if d.get("id")}
    new_ids      = {c["id"] for c in KNOWLEDGE}
    to_delete    = sorted(set(existing) - new_ids)
    no_id_count  = col.count_documents({"id": {"$exists": False}})
    changed      = [c for c in KNOWLEDGE
                    if c["id"] not in existing
                    or existing[c["id"]].get("text") != c["text"]
                    or existing[c["id"]].get("category") != c["category"]]

    by_cat = {}
    for c in KNOWLEDGE:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1

    print(f"📚 knowledge_data.py: {len(KNOWLEDGE)} chunks → {by_cat}")
    print(f"🗄️  Collection now:   {col.count_documents({})} chunks")
    print(f"   ➕ new/changed:   {len(changed)}")
    print(f"   🗑️  to delete:     {len(to_delete) + no_id_count}  {to_delete[:12]}{' ...' if len(to_delete) > 12 else ''}")

    if dry_run:
        print("\n🔎 Dry run — nothing changed.")
        return

    # ── 1. Backup ─────────────────────────────────────────────────
    if not no_backup and col.count_documents({}) > 0:
        name = f"knowledge_chunks_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        docs = list(col.find({}))
        db[name].insert_many(docs)
        print(f"💾 Backup: {len(docs)} chunks → {name}")

    # ── 2. Embed ──────────────────────────────────────────────────
    client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    vectors = embed_all(client, [c["text"] for c in KNOWLEDGE])

    # ── 3. Upsert ─────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    ops = [
        UpdateOne(
            {"id": c["id"]},
            {"$set": {
                "id":         c["id"],
                "text":       c["text"],
                "category":   c["category"],
                "source":     SOURCE_TAG,
                "embedding":  v,
                "updated_at": now,
            }},
            upsert=True,
        )
        for c, v in zip(KNOWLEDGE, vectors)
    ]
    res = col.bulk_write(ops, ordered=False)
    print(f"✅ Upserted: {res.upserted_count} new, {res.modified_count} updated")

    # ── 4. Delete stale chunks ────────────────────────────────────
    deleted = col.delete_many({"$or": [{"id": {"$nin": list(new_ids)}}, {"id": {"$exists": False}}]}).deleted_count
    print(f"🗑️  Deleted {deleted} stale chunks")

    print(f"🏁 Done — collection now has {col.count_documents({})} chunks. "
          "Atlas re-indexes vectors automatically (usually within a minute).")


if __name__ == "__main__":
    main()
