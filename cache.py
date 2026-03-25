import hashlib
import json
import os
import time
import config

CACHE_DIR = ".query_cache"
TTL_SECONDS = config.cache_ttl


def _key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def get(query: str) -> dict | None:
    path = _path(_key(query))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        entry = json.load(f)
    if time.time() - entry["cached_at"] > TTL_SECONDS:
        os.remove(path)
        return None
    return entry["data"]


def set(query: str, data: dict):
    path = _path(_key(query))
    with open(path, "w") as f:
        json.dump({"cached_at": time.time(), "data": data}, f)


def invalidate(query: str):
    path = _path(_key(query))
    if os.path.exists(path):
        os.remove(path)


def clear_all():
    if os.path.exists(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, f))
