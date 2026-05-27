from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .train import run_from_dict, run_cli


@dataclass
class DisentangleV7API:
    """Programmatic entrypoint for V7.

    Minimal wrapper today; later we can add disentanglement-related knobs here.
    """

    defaults: Dict[str, Any] = field(default_factory=dict)

    def finetune(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        cfg: Dict[str, Any] = dict(self.defaults)
        if overrides:
            cfg.update(overrides)
        run_from_dict(cfg)

    def cli(self, argv: Optional[list[str]] = None) -> None:
        run_cli(argv)

