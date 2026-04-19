"""In-process synchronous event bus with typed publish/subscribe."""

import logging
from typing import Callable

from f1_commentary.schemas.events import BaseEvent, EventType

logger = logging.getLogger(__name__)


class EventBus:
    """Synchronous in-process event bus.

    Supports both type-specific and global (wildcard) subscriptions.
    All dispatch is synchronous — callbacks are invoked in registration order.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[BaseEvent], None]]] = {}
        self._global_subscribers: list[Callable[[BaseEvent], None]] = []

    def subscribe(
        self, event_type: EventType, callback: Callable[[BaseEvent], None]
    ) -> None:
        """Register *callback* to be called when an event of *event_type* is published."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[BaseEvent], None]) -> None:
        """Register *callback* to receive every published event regardless of type."""
        self._global_subscribers.append(callback)

    def publish(self, event: BaseEvent) -> None:
        """Publish *event* to type-specific and global subscribers."""
        logger.debug(
            "Publishing %s (id=%s)", event.event_type.value, event.event_id
        )
        # Type-specific subscribers
        for cb in self._subscribers.get(event.event_type, []):
            cb(event)
        # Global subscribers
        for cb in self._global_subscribers:
            cb(event)

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
        self._global_subscribers.clear()
