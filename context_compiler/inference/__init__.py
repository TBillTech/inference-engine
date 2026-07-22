"""Resolution sub-package: provider interface and concrete implementations."""

from context_compiler.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
    # Backward-compatible aliases
    InferenceProvider,
    InferenceRequest,
    InferenceResponse,
)
from context_compiler.inference.mock_provider import MockProvider

__all__ = [
    "ResolutionProvider",
    "ResolutionRequest",
    "ResolutionResult",
    # Backward-compatible aliases
    "InferenceProvider",
    "InferenceRequest",
    "InferenceResponse",
    "MockProvider",
]
