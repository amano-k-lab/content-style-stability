"""Framework-independent style-transfer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class StyleTransfererPort(ABC):
    """Interface for a tensor-to-tensor style-transfer implementation."""

    @abstractmethod
    def transfer(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return a batched RGB tensor in the ``[0, 1]`` range."""

        raise NotImplementedError
