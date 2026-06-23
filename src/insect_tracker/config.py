from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Apply CLI-style overrides like source_type='webcam' or detector_weights='x.pt'."""
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, field = key.partition("_")
        if section in config and field in config[section]:
            config[section][field] = value
    return config
