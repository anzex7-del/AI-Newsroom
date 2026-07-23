"""Load News Scout source configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_sources(path: Path) -> list[dict[str, Any]]:
    """Load the source list from a YAML file."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")

    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ValueError("'sources' must be a list")

    return [source for source in sources if isinstance(source, dict)]
