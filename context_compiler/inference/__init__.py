"""Resolution sub-package: provider interface, strategy interface, and concrete implementations."""

from context_compiler.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
)
from context_compiler.inference.strategy import ResolutionStrategy, PromptStrategy
from context_compiler.inference.mock_provider import MockProvider

__all__ = [
    "ResolutionProvider",
    "ResolutionRequest",
    "ResolutionResult",
    "MockProvider",
    "ResolutionStrategy",
    "PromptStrategy",
]
