"""
FastAPI Dependencies
====================
Dependency injection functions for FastAPI endpoints.
"""

from typing import Annotated
from functools import lru_cache
from fastapi import Depends

from api.config import Settings
from api.services.orchestrator_adapter import OrchestratorAdapter, get_orchestrator_adapter


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings dependency.
    Returns singleton Settings instance.
    """
    return Settings()


def get_orchestrator() -> OrchestratorAdapter:
    """
    Dependency to get the orchestrator adapter singleton.
    Can be overridden in tests.
    """
    return get_orchestrator_adapter()


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
OrchestratorDep = Annotated[OrchestratorAdapter, Depends(get_orchestrator)]
