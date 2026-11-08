from functools import cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from rein.schema import Message

MIXED_PT_EN = "pt-BR+en"
MIN_SECONDARY_SHARE = 0.1
LANGUAGE_BY_CODE: dict[str, Language] = {
    "en": Language.ENGLISH,
    "pt-BR": Language.PORTUGUESE,
    "es": Language.SPANISH,
    "fr": Language.FRENCH,
    "de": Language.GERMAN,
    "it": Language.ITALIAN,
    "ja": Language.JAPANESE,
    "zh": Language.CHINESE,
    "ko": Language.KOREAN,
    "ru": Language.RUSSIAN,
    "ar": Language.ARABIC,
    "hi": Language.HINDI,
}


@cache
def _detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(*LANGUAGE_BY_CODE.values()).build()


def _spoken_text(messages: tuple[Message, ...]) -> str:
    return "\n".join(message.text for message in messages if message.role in {"user", "assistant"})


def matches_language(messages: tuple[Message, ...], code: str) -> bool:
    """Whether the spoken turns are written in the requested language (both languages for the PT-EN mix)."""
    text = _spoken_text(messages)
    if not text.strip():
        return False
    detector = _detector()
    if code == MIXED_PT_EN:
        pair = {Language.PORTUGUESE, Language.ENGLISH}
        shares = {value.language: value.value for value in detector.compute_language_confidence_values(text)}
        line_languages = {detector.detect_language_of(line) for line in text.splitlines() if line.strip()}
        dominant_in_pair = detector.detect_language_of(text) in pair
        both_present = pair <= line_languages or min(shares.get(lang, 0.0) for lang in pair) >= MIN_SECONDARY_SHARE
        return dominant_in_pair and both_present
    return detector.detect_language_of(text) == LANGUAGE_BY_CODE[code]
