"""
Orchestrator Bridge

Provides thread-safe communication between the synchronous
orchestrator thread and the async WebSocket broadcaster.
"""

import asyncio
import logging
import threading
from queue import Queue, Empty, Full
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class OrchestratorBridge:
    """
    Thread-safe bridge between orchestrator and async broadcaster.

    The orchestrator runs in a separate thread and produces data.
    This bridge safely transfers that data to the async context
    for WebSocket broadcasting.
    """

    _instance: Optional["OrchestratorBridge"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._queue: Queue[Dict[str, Any]] = Queue(maxsize=50)
        self._transfer_task: Optional[asyncio.Task] = None
        self._running = False
        self._broadcaster = None

    @classmethod
    def get_instance(cls) -> "OrchestratorBridge":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            cls._instance = None

    def push_sync(self, data: Dict[str, Any]) -> bool:
        """
        Push data from the orchestrator thread (synchronous).

        This method is called from the orchestrator's _tick() loop.
        It's thread-safe and non-blocking.
        
        Returns:
            True if data was queued successfully
        """
        try:
            # Non-blocking put
            self._queue.put_nowait(data)
            return True
        except Full:
            # Queue full - drop oldest and retry
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(data)
                return True
            except Empty:
                return False

    async def start_transfer(self, broadcaster) -> None:
        """
        Start the async transfer loop.

        This runs in the asyncio event loop and transfers data
        from the thread-safe queue to the async broadcaster.
        """
        self._broadcaster = broadcaster
        self._running = True
        self._transfer_task = asyncio.create_task(self._transfer_loop())
        logger.info("OrchestratorBridge transfer started")

    async def stop_transfer(self) -> None:
        """Stop the transfer loop."""
        self._running = False
        if self._transfer_task:
            self._transfer_task.cancel()
            try:
                await self._transfer_task
            except asyncio.CancelledError:
                pass
        self._broadcaster = None
        logger.info("OrchestratorBridge transfer stopped")

    async def _transfer_loop(self) -> None:
        """Transfer data from sync queue to async broadcaster."""
        while self._running:
            try:
                # Check sync queue frequently
                await asyncio.sleep(0.01)  # 100Hz check rate

                # Drain all available messages
                messages_transferred = 0
                while messages_transferred < 10:  # Max 10 per cycle
                    try:
                        data = self._queue.get_nowait()
                        if self._broadcaster:
                            await self._broadcaster.queue_update(data)
                        messages_transferred += 1
                    except Empty:
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Transfer error: {e}")
                await asyncio.sleep(0.1)

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


# Global singleton accessor
def get_orchestrator_bridge() -> OrchestratorBridge:
    """Get the singleton bridge instance."""
    return OrchestratorBridge.get_instance()


# Global function for orchestrator to call
def push_to_websocket(data: Dict[str, Any]) -> bool:
    """
    Push data to WebSocket clients.

    This is the function that the orchestrator calls after each tick.
    It's thread-safe and can be called from any thread.
    
    Returns:
        True if data was queued successfully
    """
    bridge = OrchestratorBridge.get_instance()
    return bridge.push_sync(data)
