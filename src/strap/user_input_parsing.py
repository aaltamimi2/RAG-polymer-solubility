"""Shared parsing helpers for messy user-entered paths and temperatures."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import re

from strap.tools._helpers import normalize_wsl_path


_OUTPUT_COMMAND_RE = re.compile(
    r"\b(?:save|saved|write|written|store|stored|output|export|put|place|create|generate)\b"
    r"[\s\S]{0,140}?\b(?:to|under|in|at)\s+",
    re.IGNORECASE,
)
_WRAPPED_PATH_BREAK_RE = re.compile(
    r"(?P<prefix>(?:\\\\|//|/|~|[A-Za-z]:[\\/])[\w./\\:~$-]*)\r?\n[ \t]*(?P<fragment>[\w./\\:~$-]+)"
)
_UNQUOTED_PATH_RE = re.compile(r"(?P<path>(?:\\\\|//|/|~|[A-Za-z]:[\\/])\S+)")
_PATH_FRAGMENT_STOPWORDS = {
    "also",
    "and",
    "but",
    "except",
    "excluding",
    "include",
    "including",
    "next",
    "not",
    "only",
    "or",
    "please",
    "show",
    "then",
    "with",
    "without",
}
_DEFAULT_OUTPUT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".html",
    ".htm",
    ".json",
    ".csv",
    ".xlsx",
    ".txt",
    ".md",
}


@dataclass(frozen=True)
class OutputDestination:
    """Normalized destination parsed from a user request."""

    output_dir: str
    filename_hint: str | None = None


@dataclass(frozen=True)
class TemperatureMention:
    """Temperature mention normalized to Celsius."""

    value_c: float
    raw_value: float
    raw_unit: str
    normalized_unit: str
    start: int
    end: int


_UNIT_ALIASES = {
    "c": "celsius",
    "centigrade": "celsius",
    "celsius": "celsius",
    "celcius": "celsius",
    "celsuis": "celsius",
    "degreec": "celsius",
    "degreesc": "celsius",
    "degc": "celsius",
    "f": "fahrenheit",
    "fahrenheit": "fahrenheit",
    "fahrenehit": "fahrenheit",
    "farenheit": "fahrenheit",
    "farhenheit": "fahrenheit",
    "degf": "fahrenheit",
    "degreef": "fahrenheit",
    "degreesf": "fahrenheit",
    "k": "kelvin",
    "kelvin": "kelvin",
    "kelvn": "kelvin",
    "kevlin": "kelvin",
    "degk": "kelvin",
}
_FUZZY_UNITS = ("celsius", "fahrenheit", "kelvin")
_TEMP_MENTION_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>"
    r"°\s*[CFK]|"
    r"(?:deg(?:ree|rees?)?|degrees?)\s*(?:[CFK]|celsius|celcius|celsuis|centigrade|fahrenheit|fahrenehit|farenheit|farhenheit|kelvin|kelvn|kevlin)|"
    r"[CFK]\b|"
    r"celsius|celcius|celsuis|centigrade|"
    r"fahrenheit|fahrenehit|farenheit|farhenheit|"
    r"kelvin|kelvn|kevlin"
    r")\b",
    re.IGNORECASE,
)
_TEMP_CEILING_CUE_RE = re.compile(
    r"\b(below|under|up\s+to|at\s+or\s+below|less\s+than|max(?:imum)?|no\s+more\s+than|not\s+above)\b|<=",
    re.IGNORECASE,
)


def _looks_like_path_fragment(prefix: str, fragment: str, following: str) -> bool:
    fragment = str(fragment or "").strip()
    if not fragment:
        return False
    lowered = fragment.lower()
    if lowered in _PATH_FRAGMENT_STOPWORDS:
        return False
    if fragment[0].isupper() and not re.search(r"[/\\.:~$-]", fragment):
        return False
    if re.search(r"[/\\.:~$]", fragment):
        return True
    if fragment.startswith(("-", "_")):
        return True
    if str(prefix or "").endswith(("/", "\\", "-", "_", ".")):
        return True
    if len(fragment) <= 4 and re.match(
        r"\r?\n[ \t]*(?:[-_.\\/]|[\w$~.-]+[/\\]|[\w$~-]+\.)",
        following or "",
    ):
        return True
    return False


def _repair_wrapped_path_text(text: str) -> str:
    repaired = str(text or "")
    while True:
        def _join_if_path_fragment(match: re.Match[str]) -> str:
            prefix = match.group("prefix")
            fragment = match.group("fragment")
            following = match.string[match.end() :]
            if _looks_like_path_fragment(prefix, fragment, following):
                return f"{prefix}{fragment}"
            return match.group(0)

        next_value = _WRAPPED_PATH_BREAK_RE.sub(_join_if_path_fragment, repaired)
        if next_value == repaired:
            return repaired
        repaired = next_value


def _clean_unquoted_path(raw_path: str) -> str:
    candidate = _repair_wrapped_path_text(raw_path).strip().strip("\"'`")
    candidate = re.split(r"(?<!\.)[;,](?:\s|$)", candidate, maxsplit=1)[0]
    candidate = re.split(r"\.(?:\s|$)", candidate, maxsplit=1)[0] if candidate.endswith(".") else candidate
    return candidate.strip().rstrip(".,;")


def split_output_destination(
    raw_path: str,
    *,
    output_extensions: set[str] | None = None,
) -> OutputDestination | None:
    """Normalize a raw path and split file hints from directory paths."""
    cleaned = _clean_unquoted_path(raw_path)
    if not cleaned:
        return None
    normalized = normalize_wsl_path(cleaned)
    path = Path(normalized).expanduser()
    extensions = output_extensions or _DEFAULT_OUTPUT_EXTENSIONS
    if path.suffix.lower() in extensions:
        return OutputDestination(output_dir=str(path.parent), filename_hint=path.name)
    return OutputDestination(output_dir=str(path), filename_hint=None)


def extract_output_destination(
    query: str,
    *,
    output_extensions: set[str] | None = None,
) -> OutputDestination | None:
    """Extract a requested output path, repairing common CLI line wrapping."""
    if not query:
        return None

    for command in _OUTPUT_COMMAND_RE.finditer(query):
        tail = query[command.end() :]

        quoted = re.match(r"(?P<quote>[\"'`])(?P<path>.+?)(?P=quote)", tail, re.S)
        if quoted:
            return split_output_destination(
                quoted.group("path"),
                output_extensions=output_extensions,
            )

        repaired_tail = _repair_wrapped_path_text(tail)
        unquoted = _UNQUOTED_PATH_RE.match(repaired_tail.strip())
        if not unquoted:
            continue
        destination = split_output_destination(
            unquoted.group("path"),
            output_extensions=output_extensions,
        )
        if destination is not None:
            return destination
    return None


def _normalize_temperature_unit(raw_unit: str) -> str | None:
    unit = re.sub(r"[^A-Za-z]+", "", str(raw_unit or "")).lower()
    for prefix in ("degrees", "degree", "deg"):
        if unit.startswith(prefix) and len(unit) > len(prefix):
            unit = unit[len(prefix) :]
            break
    if not unit:
        return None
    if unit in _UNIT_ALIASES:
        return _UNIT_ALIASES[unit]
    match = get_close_matches(unit, _FUZZY_UNITS, n=1, cutoff=0.78)
    return match[0] if match else None


def _to_celsius(value: float, normalized_unit: str) -> float:
    if normalized_unit == "fahrenheit":
        return (value - 32.0) * 5.0 / 9.0
    if normalized_unit == "kelvin":
        return value - 273.15
    return value


def extract_temperature_mentions_c(text: str) -> list[TemperatureMention]:
    """Extract temperature mentions and convert supported units to Celsius."""
    mentions: list[TemperatureMention] = []
    for match in _TEMP_MENTION_RE.finditer(text or ""):
        normalized_unit = _normalize_temperature_unit(match.group("unit"))
        if normalized_unit is None:
            continue
        raw_value = float(match.group("value"))
        mentions.append(
            TemperatureMention(
                value_c=_to_celsius(raw_value, normalized_unit),
                raw_value=raw_value,
                raw_unit=match.group("unit"),
                normalized_unit=normalized_unit,
                start=match.start(),
                end=match.end(),
            )
        )
    return mentions


def extract_temperatures_c(text: str) -> list[float]:
    return [mention.value_c for mention in extract_temperature_mentions_c(text)]


def contains_temperature_mention(text: str) -> bool:
    return bool(extract_temperature_mentions_c(text))


def has_temperature_ceiling(text: str) -> bool:
    if not contains_temperature_mention(text):
        return False
    return bool(_TEMP_CEILING_CUE_RE.search(text or ""))


def last_temperature_c(text: str) -> float | None:
    mentions = extract_temperature_mentions_c(text)
    return mentions[-1].value_c if mentions else None
