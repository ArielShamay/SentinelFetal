# -*- coding: utf-8 -*-
"""
Persistence Manager for K-of-N Alert Smoothing.

This module implements a stateful buffer that tracks the last N window decisions
and enforces a K-of-N rule before issuing final alerts. This prevents flickering
and false positives caused by isolated transient spikes.

References:
    SentinelFetal Stage 4 Documentation - Persistence Mechanism
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersistenceState:
    """
    State tracker for K-of-N persistence logic.
    
    Attributes:
        K: Minimum number of windows that must be above threshold.
        N: Total size of the sliding window buffer.
        buffer: Circular buffer holding the last N boolean decisions.
        alert_active: Whether an alert is currently active.
    """
    K: int
    N: int
    buffer: deque = field(default_factory=deque)
    alert_active: bool = False
    
    def __post_init__(self):
        """Validate K and N parameters."""
        if self.K > self.N:
            raise ValueError(f"K ({self.K}) cannot be greater than N ({self.N})")
        if self.K < 1 or self.N < 1:
            raise ValueError(f"K and N must be >= 1, got K={self.K}, N={self.N}")
        
        # Initialize buffer with False values
        self.buffer = deque([False] * self.N, maxlen=self.N)


class PersistenceManager:
    """
    Manages persistence logic for multiple monitored entities.
    
    The manager maintains separate state for each monitor ID (e.g., patient ID)
    and applies K-of-N smoothing to prevent alert flickering.
    """
    
    def __init__(self, K: int = 2, N: int = 3):
        """
        Initialize the persistence manager.
        
        Args:
            K: Minimum windows above threshold to trigger alert (default: 2).
            N: Size of the window buffer (default: 3).
        """
        self.K = K
        self.N = N
        self.states: dict[str, PersistenceState] = {}
        
        logger.info(f"Initialized PersistenceManager with K={K}, N={N}")
    
    def update(
        self,
        monitor_id: str,
        above_threshold: bool
    ) -> bool:
        """
        Update the persistence state and return final alert decision.
        
        Args:
            monitor_id: Unique identifier for the monitored entity.
            above_threshold: Whether the current window is above threshold.
            
        Returns:
            True if an alert should be raised (K-of-N condition met), False otherwise.
        """
        # Initialize state if new monitor
        if monitor_id not in self.states:
            self.states[monitor_id] = PersistenceState(K=self.K, N=self.N)
        
        state = self.states[monitor_id]
        
        # Add new decision to buffer (oldest is auto-dropped due to maxlen)
        state.buffer.append(above_threshold)
        
        # Count how many of the last N windows were above threshold
        count_above = sum(state.buffer)
        
        # Apply K-of-N logic
        should_alert = count_above >= self.K
        
        # Update alert state
        state.alert_active = should_alert
        
        logger.debug(
            f"Monitor {monitor_id}: {count_above}/{self.N} above threshold "
            f"-> Alert: {should_alert}"
        )
        
        return should_alert
    
    def get_state(self, monitor_id: str) -> Optional[PersistenceState]:
        """
        Get the current persistence state for a monitor.
        
        Args:
            monitor_id: Monitor identifier.
            
        Returns:
            PersistenceState if monitor exists, None otherwise.
        """
        return self.states.get(monitor_id)
    
    def reset(self, monitor_id: Optional[str] = None) -> None:
        """
        Reset persistence state.
        
        Args:
            monitor_id: Specific monitor to reset. If None, resets all.
        """
        if monitor_id is None:
            logger.info("Resetting all persistence states")
            self.states.clear()
        elif monitor_id in self.states:
            logger.info(f"Resetting persistence state for {monitor_id}")
            del self.states[monitor_id]
    
    def get_buffer_list(self, monitor_id: str) -> List[bool]:
        """
        Get the current buffer as a list (for debugging/visualization).
        
        Args:
            monitor_id: Monitor identifier.
            
        Returns:
            List of boolean decisions, or empty list if monitor not found.
        """
        state = self.states.get(monitor_id)
        return list(state.buffer) if state else []


def test_persistence():
    """Simple test function to verify K-of-N logic."""
    manager = PersistenceManager(K=2, N=3)
    
    # Test sequence: [False, True, True] -> should trigger on 3rd
    assert manager.update("test", False) == False  # 0/3
    assert manager.update("test", True) == False   # 1/3
    assert manager.update("test", True) == True    # 2/3 ✓
    
    # Test sequence continues: [True, True, False] -> still 2/3
    assert manager.update("test", False) == True   # 2/3 ✓
    
    # Test sequence: [True, False, False] -> drops to 1/3
    assert manager.update("test", False) == False  # 1/3
    
    logger.info("✓ Persistence tests passed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_persistence()
