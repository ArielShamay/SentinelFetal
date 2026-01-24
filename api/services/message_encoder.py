"""
Message Encoder

Handles encoding/decoding of WebSocket messages using MessagePack
for performance, with JSON fallback.
"""

import json
from typing import Any, Dict, Optional

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


def encode_message(data: Dict[str, Any], format: str = "json") -> bytes:
    """
    Encode a message for WebSocket transmission.

    Args:
        data: The data to encode
        format: 'msgpack' or 'json' (default)

    Returns:
        Encoded bytes
    """
    if format == "msgpack" and MSGPACK_AVAILABLE:
        return msgpack.packb(data, default=str, use_bin_type=True)

    return json.dumps(data, default=str).encode("utf-8")


def decode_message(data: bytes, format: str = "json") -> Dict[str, Any]:
    """
    Decode a WebSocket message.

    Args:
        data: The raw bytes
        format: 'msgpack' or 'json' (default)

    Returns:
        Decoded dictionary
    """
    if format == "msgpack" and MSGPACK_AVAILABLE:
        return msgpack.unpackb(data, raw=False)

    return json.loads(data.decode("utf-8"))


def negotiate_format(accept_header: Optional[str] = None) -> str:
    """
    Negotiate message format based on Accept header or preferences.

    Args:
        accept_header: Client's Accept header value

    Returns:
        'msgpack' or 'json'
    """
    if accept_header:
        if "application/msgpack" in accept_header and MSGPACK_AVAILABLE:
            return "msgpack"
    
    # Default to JSON for broader compatibility
    return "json"


def is_msgpack_available() -> bool:
    """Check if msgpack is available."""
    return MSGPACK_AVAILABLE
