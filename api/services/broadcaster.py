"""
AsyncBroadcaster

Manages WebSocket connections and broadcasts patient updates
to all connected clients efficiently.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Set, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""
    client_id: str
    websocket: WebSocket
    subscribed_patients: Set[str] = field(default_factory=set)  # Empty = all
    connected_at: float = field(default_factory=time.time)
    last_pong: float = field(default_factory=time.time)
    message_count: int = 0
    is_alive: bool = True
    message_format: str = "json"  # "json" or "msgpack"


class AsyncBroadcaster:
    """
    Async WebSocket broadcaster with topic-based subscriptions.

    Features:
    - Efficient multi-client broadcast
    - Patient-based subscription filtering
    - Automatic dead connection cleanup
    - Backpressure handling for slow clients
    """

    _instance: Optional["AsyncBroadcaster"] = None

    def __init__(self):
        self._clients: Dict[str, ClientConnection] = {}
        self._lock = asyncio.Lock()
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._running = False
        self._broadcast_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @classmethod
    def get_instance(cls) -> "AsyncBroadcaster":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        cls._instance = None

    async def start(self) -> None:
        """Start the broadcaster background tasks."""
        if self._running:
            return

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("AsyncBroadcaster started")

    async def stop(self) -> None:
        """Stop the broadcaster and disconnect all clients."""
        self._running = False

        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            for client in self._clients.values():
                try:
                    await client.websocket.close()
                except Exception:
                    pass
            self._clients.clear()

        logger.info("AsyncBroadcaster stopped")

    async def register(
        self, 
        websocket: WebSocket, 
        client_id: Optional[str] = None,
        message_format: str = "json",
    ) -> str:
        """
        Register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Optional existing client ID for reconnection
            message_format: "json" or "msgpack"

        Returns:
            The assigned client ID
        """
        if client_id is None or client_id not in self._clients:
            client_id = str(uuid.uuid4())[:8]

        client = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            message_format=message_format,
        )

        async with self._lock:
            self._clients[client_id] = client

        logger.info(f"Client {client_id} registered. Total: {len(self._clients)}")
        return client_id

    async def unregister(self, client_id: str) -> None:
        """Unregister a client connection."""
        async with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"Client {client_id} unregistered. Total: {len(self._clients)}")

    async def subscribe(self, client_id: str, patient_ids: Set[str]) -> None:
        """
        Subscribe a client to specific patients.

        Args:
            client_id: The client ID
            patient_ids: Set of patient IDs to subscribe to (empty = all)
        """
        async with self._lock:
            if client_id in self._clients:
                self._clients[client_id].subscribed_patients = patient_ids
                logger.debug(f"Client {client_id} subscribed to {patient_ids or 'all'}")

    async def queue_update(self, data: Dict[str, Any]) -> None:
        """
        Queue an update for broadcast.

        Called from the orchestrator bridge.
        """
        # DEBUG: Log first few queues
        if not hasattr(self, '_queue_count'):
            self._queue_count = 0
        self._queue_count += 1
        
        if self._queue_count <= 10:
            patient_id = data.get('patient_id', 'unknown')
            logger.info(f"📻 BROADCASTER: Queuing {patient_id}, count={self._queue_count}, queue={self._message_queue.qsize()}")
        
        try:
            # Non-blocking put with timeout
            await asyncio.wait_for(
                self._message_queue.put(data),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Queue full - drop oldest message
            try:
                self._message_queue.get_nowait()
                await self._message_queue.put(data)
            except asyncio.QueueEmpty:
                pass

    def queue_update_sync(self, data: Dict[str, Any]) -> None:
        """Thread-safe synchronous queue update."""
        try:
            self._message_queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    async def _broadcast_loop(self) -> None:
        """Main broadcast loop - sends queued messages to all clients."""
        while self._running:
            try:
                # Wait for next message
                data = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                # Broadcast to all clients
                await self._broadcast(data)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected clients."""
        from api.services.message_encoder import encode_message

        # DEBUG: Log first few broadcasts
        if not hasattr(self, '_broadcast_count'):
            self._broadcast_count = 0
        self._broadcast_count += 1
        
        # Log every 100 broadcasts to see activity
        if self._broadcast_count % 100 == 0:
            patient_id = data.get('patient_id', 'unknown')
            logger.info(f"📣 _broadcast #{self._broadcast_count}: Sending {patient_id}, clients={len(self._clients)}")

        dead_clients: List[str] = []

        async with self._lock:
            clients = list(self._clients.items())

        if not clients:
            # Log periodically if no clients
            if self._broadcast_count % 100 == 0:
                logger.warning(f"❌ No clients connected at broadcast #{self._broadcast_count}")
            return

        for client_id, client in clients:
            if not client.is_alive:
                dead_clients.append(client_id)
                continue

            try:
                # Filter patients if subscribed
                filtered_data = self._filter_for_client(data, client)
                if filtered_data is None:
                    continue

                # Encode and send as JSON text
                import json
                import numpy as np
                
                # Custom encoder to handle numpy arrays
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                # Convert numpy arrays recursively
                filtered_data = convert_numpy(filtered_data)
                
                message = json.dumps(filtered_data)
                await asyncio.wait_for(
                    client.websocket.send_text(message),
                    timeout=0.5
                )
                client.message_count += 1
                
                # DEBUG: Log first few sends AND every 100
                if self._broadcast_count <= 5 or self._broadcast_count % 100 == 0:
                    logger.info(f"✉️ Sent to {client_id}: {len(message)} bytes")
                    if self._broadcast_count <= 2:
                        # Log message structure for first 2 messages
                        logger.info(f"📦 Message structure: {list(filtered_data.keys())}")
                        logger.info(f"📊 Message type: {filtered_data.get('type', 'unknown')}")
                        logger.info(f"👤 Patient ID: {filtered_data.get('patient_id', 'unknown')}")

            except asyncio.TimeoutError:
                logger.warning(f"Client {client_id} too slow, marking dead")
                client.is_alive = False
                dead_clients.append(client_id)
            except Exception as e:
                logger.warning(f"Error sending to {client_id}: {e}")
                client.is_alive = False
                dead_clients.append(client_id)

        # Clean up dead clients
        for client_id in dead_clients:
            await self.unregister(client_id)

    def _filter_for_client(
        self,
        data: Dict[str, Any],
        client: ClientConnection,
    ) -> Optional[Dict[str, Any]]:
        """Filter patient data based on client subscription."""
        if not client.subscribed_patients:
            # No filter - send all
            return data

        # Filter patients list
        if "patients" in data:
            filtered_patients = [
                p for p in data["patients"]
                if p.get("patient_id") in client.subscribed_patients
            ]
            if not filtered_patients:
                return None
            return {**data, "patients": filtered_patients}

        # Single patient update
        if "patient_id" in data:
            if data["patient_id"] not in client.subscribed_patients:
                return None

        return data

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to detect dead connections."""
        from api.services.message_encoder import encode_message

        while self._running:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds

                dead_clients: List[str] = []

                async with self._lock:
                    clients = list(self._clients.items())

                for client_id, client in clients:
                    # Check for missed pongs
                    if time.time() - client.last_pong > 90:
                        logger.warning(f"Client {client_id} missed 3 pongs, disconnecting")
                        dead_clients.append(client_id)
                        continue

                    try:
                        ping_message = encode_message({
                            "type": "ping",
                            "timestamp": time.time()
                        }, client.message_format)
                        await client.websocket.send_bytes(ping_message)
                    except Exception:
                        dead_clients.append(client_id)

                for client_id in dead_clients:
                    await self.unregister(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def handle_pong(self, client_id: str) -> None:
        """Handle pong response from client."""
        async with self._lock:
            if client_id in self._clients:
                self._clients[client_id].last_pong = time.time()

    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            "connected_clients": len(self._clients),
            "queue_size": self._message_queue.qsize(),
            "running": self._running,
        }


# Convenience function
def get_broadcaster() -> AsyncBroadcaster:
    """Get the singleton broadcaster instance."""
    return AsyncBroadcaster.get_instance()
