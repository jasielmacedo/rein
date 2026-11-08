import re
from collections.abc import Iterable, Mapping
from typing import Any

from rein.genview import constants as c
from rein.schema.constants import ID_PATTERN

_TOOL_NAME = re.compile(ID_PATTERN)


def tool_specs(names: Iterable[str], descriptions: Mapping[str, str]) -> tuple[dict[str, Any], ...]:
    """Minimal function specs for the chat template; names that are not plain identifiers are left out."""
    return tuple(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions.get(name, ""),
                "parameters": dict(c.TOOL_PARAMETERS),
            },
        }
        for name in dict.fromkeys(names)
        if _TOOL_NAME.fullmatch(name)
    )
