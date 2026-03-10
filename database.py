import json
import os
from threading import Lock

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "settings.json")
_lock = Lock()


def _ensure_file():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, "w") as f:
            json.dump({}, f)


def read_db() -> dict:
    _ensure_file()
    with _lock:
        with open(DB_PATH, "r") as f:
            return json.load(f)


def write_db(data: dict) -> None:
    _ensure_file()
    with _lock:
        with open(DB_PATH, "w") as f:
            json.dump(data, f, indent=2)


def update_record(key: str, value) -> None:
    data = read_db()
    data[key] = value
    write_db(data)
