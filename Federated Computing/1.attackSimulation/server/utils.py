# server/utils.py
import json
from queue import Queue
from typing import List, Dict, Any

# Simple SSE broadcaster
_subscribers: List[Queue] = []

def subscribe() -> Queue:
    q = Queue()
    _subscribers.append(q)
    return q

def unsubscribe(q: Queue):
    try:
        _subscribers.remove(q)
    except ValueError:
        pass

def push_event(event: Dict[str, Any]):
    payload = json.dumps(event, ensure_ascii=False)
    for q in list(_subscribers):
        q.put(payload)
