"""Resolution sub-package: provider interface, strategy interface, and concrete implementations."""

from context_resolver.inference.provider import (
    ResolutionProvider,
    ResolutionRequest,
    ResolutionResult,
)
from context_resolver.inference.strategy import ResolutionStrategy, PromptStrategy
from context_resolver.inference.mock_provider import MockProvider
from context_resolver.inference.llama_cpp_provider import LocalLlamaCppProvider

__all__ = [
    "ResolutionProvider",
    "ResolutionRequest",
    "ResolutionResult",
    "MockProvider",
    "LocalLlamaCppProvider",
    "ResolutionStrategy",
    "PromptStrategy",
]
