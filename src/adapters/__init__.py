"""
Adapters that wrap existing implementations to conform to interfaces.
These enable dependency injection without changing existing code.

Adapters provide a layer of abstraction between the interfaces and
the existing concrete implementations, allowing for easy component swapping.

Usage:
    >>> from src.adapters import BaselineAdapter, ClassifierAdapter
    >>> baseline_calc = BaselineAdapter(window_minutes=2.0)
    >>> result = baseline_calc.calculate(fhr, sampling_rate=4.0)
"""

from .rule_adapters import (
    BaselineAdapter,
    VariabilityAdapter,
    DecelerationAdapter,
    TachysystoleAdapter,
    SinusoidalAdapter,
)
from .model_adapters import (
    MomentAdapter,
    ClassifierAdapter,
    FusionAdapter,
)
from .analysis_adapters import (
    OverrideAdapter,
    AlertAdapter,
)
from .data_adapters import (
    DataLoaderAdapter,
    PreprocessorAdapter,
)
from .ensemble_adapter import (
    EnsembleClassifierAdapter,
    create_ensemble_classifier,
)

__all__ = [
    # Rule adapters
    'BaselineAdapter',
    'VariabilityAdapter',
    'DecelerationAdapter',
    'TachysystoleAdapter',
    'SinusoidalAdapter',
    # Model adapters
    'MomentAdapter',
    'ClassifierAdapter',
    'FusionAdapter',
    # Ensemble adapter (V4.0)
    'EnsembleClassifierAdapter',
    'create_ensemble_classifier',
    # Analysis adapters
    'OverrideAdapter',
    'AlertAdapter',
    # Data adapters
    'DataLoaderAdapter',
    'PreprocessorAdapter',
]
