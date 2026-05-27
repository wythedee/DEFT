from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from .finetune_main import run_cli, run_from_dict


@dataclass
class CodeBrainDownstreamAPI:
    defaults: Dict[str, Any] = field(default_factory=dict)

    def finetune(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        cfg: Dict[str, Any] = dict(self.defaults)
        if overrides:
            cfg.update(overrides)
        run_from_dict(cfg)

    def cli(self, argv: Optional[Sequence[str]] = None) -> None:
        run_cli(argv)
