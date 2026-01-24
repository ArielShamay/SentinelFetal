"""
WebSocket Routes

Full WebSocket streaming implementation for real-time patient updates.
"""

import asyncio
import logging
import time
from typing import Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState

from api.services.broadcaster import AsyncBroadcaster, get_broadcaster
from api.services.message_encoder import (
    decode_message, 
    encode_message, 
    negotiate_format,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    client_id: Optional[str] = Query(default=None),
    format: Optional[str] = Query(default=None),
):
    """
    Main WebSocket streaming endpoint.

    Connects to receive real-time patient updates for all patients.

    Query Parameters:
        client_id: Optional client ID for reconnection
        format: Message format ('json' or 'msgpack')
    """
    await websocket.accept()

    broadcaster = get_broadcaster()
    
    # Negotiate format
    accept_header = websocket.headers.get("accept", "")
    msg_format = format or negotiate_format(accept_header)
    
    assigned_id = await broadcaster.register(websocket, client_id, msg_format)

    # Send connection confirmation
    welcome = {
        "type": "connected",
        "client_id": assigned_id,
        "format": msg_format,
        "timestamp": time.time(),
    }
    await websocket.send_bytes(encode_message(welcome, msg_format))
    logger.info(f"Client {assigned_id} connected to /ws/stream")

    try:
        await _handle_client_loop(websocket, assigned_id, msg_format, broadcaster)
    except WebSocketDisconnect:
        logger.info(f"Client {assigned_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {assigned_id}: {e}")
    finally:
        await broadcaster.unregister(assigned_id)


@router.websocket("/stream/{patient_id}")
async def websocket_patient_stream(
    websocket: WebSocket,
    patient_id: str,
    client_id: Optional[str] = Query(default=None),
    format: Optional[str] = Query(default=None),
):
    """
    Single patient WebSocket stream.

    Connects to receive updates for a specific patient only.
    """
    await websocket.accept()

    broadcaster = get_broadcaster()
    
    # Negotiate format
    accept_header = websocket.headers.get("accept", "")
    msg_format = format or negotiate_format(accept_header)
    
    assigned_id = await broadcaster.register(websocket, client_id, msg_format)

    # Subscribe to specific patient
    await broadcaster.subscribe(assigned_id, {patient_id})

    # Send connection confirmation
    welcome = {
        "type": "connected",
        "client_id": assigned_id,
        "subscribed_to": patient_id,
        "format": msg_format,
        "timestamp": time.time(),
    }
    await websocket.send_bytes(encode_message(welcome, msg_format))
    logger.info(f"Client {assigned_id} connected to /ws/stream/{patient_id}")

    try:
        await _handle_client_loop(websocket, assigned_id, msg_format, broadcaster)
    except WebSocketDisconnect:
        logger.info(f"Client {assigned_id} disconnected from patient {patient_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {assigned_id}: {e}")
    finally:
        await broadcaster.unregister(assigned_id)


async def _handle_client_loop(
    websocket: WebSocket,
    client_id: str,
    msg_format: str,
    broadcaster: AsyncBroadcaster,
) -> None:
    """Handle incoming messages from client."""
    while True:
        try:
            # Wait for message with timeout
            data = await asyncio.wait_for(
                websocket.receive_bytes(),
                timeout=60.0
            )
            message = decode_message(data, msg_format)
            await _handle_client_message(client_id, message, broadcaster)

        except asyncio.TimeoutError:
            # No message received, check connection
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            # Send ping to keep alive
            try:
                ping = encode_message({"type": "ping", "timestamp": time.time()}, msg_format)
                await websocket.send_bytes(ping)
            except Exception:
                break


async def _handle_client_message(
    client_id: str,
    message: dict,
    broadcaster: AsyncBroadcaster,
) -> None:
    """Handle incoming messages from clients."""
    msg_type = message.get("type")

    if msg_type == "pong":
        await broadcaster.handle_pong(client_id)

    elif msg_type == "subscribe":
        patient_ids = set(message.get("patient_ids", []))
        await broadcaster.subscribe(client_id, patient_ids)
        logger.debug(f"Client {client_id} subscribed to {patient_ids}")

    elif msg_type == "unsubscribe":
        await broadcaster.subscribe(client_id, set())  # Clear subscription
        logger.debug(f"Client {client_id} unsubscribed from all")

    elif msg_type == "ping":
        # Client sent ping, respond with pong (already handled by heartbeat)
        pass

    else:
        logger.warning(f"Unknown message type from {client_id}: {msg_type}")


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket broadcaster statistics."""
    broadcaster = get_broadcaster()
    return broadcaster.get_stats()


@router.get("/health")
async def websocket_health():
    """WebSocket subsystem health check."""
    broadcaster = get_broadcaster()
    stats = broadcaster.get_stats()
    
    return {
        "status": "healthy" if stats["running"] else "stopped",
        "clients": stats["connected_clients"],
        "queue_size": stats["queue_size"],
    }
