"""
UI Package - DEPRECATED.

The SentinelFetal UI has migrated to React (frontend/) + FastAPI (api/).
This package is maintained for backwards compatibility only.

State Bridge components have been moved to src/interfaces/state_bridge.py

For new code, import from:
    from src.interfaces.state_bridge import (
        get_data_bridge,
        DataBridge,
        PatientSnapshot,
        WardSnapshot,
        HighlightRegion,
        create_snapshot_from_pipeline_result,
    )
"""

from pathlib import Path

UI_DIR = Path(__file__).parent

# Re-export state_bridge components for backwards compatibility
from src.interfaces.state_bridge import (
    get_data_bridge,
    reset_data_bridge,
    DataBridge,
    PatientSnapshot,
    WardSnapshot,
    HighlightRegion,
    create_snapshot_from_pipeline_result,
)

__all__ = [
    'UI_DIR',
    # State Bridge (from src.interfaces)
    'get_data_bridge',
    'reset_data_bridge',
    'DataBridge',
    'PatientSnapshot',
    'WardSnapshot',
    'HighlightRegion',
    'create_snapshot_from_pipeline_result',
]
