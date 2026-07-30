import asyncio
from collections import defaultdict
from typing import Any


class EventBus:
    """In-process async pub/sub used to fan out job/GPU events to WebSocket clients."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, topic: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers[topic].add(queue)
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        self._subscribers[topic].discard(queue)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        for queue in list(self._subscribers.get(topic, ())):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(payload)


event_bus = EventBus()
