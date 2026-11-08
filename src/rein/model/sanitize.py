import re

from transformers import PreTrainedTokenizerBase

from rein.model import constants as c


class TokenSanitizer:
    """Rewrites every added-token string in a text so it tokenizes as plain text, never as a special token."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        added = sorted(tokenizer.get_added_vocab(), key=len, reverse=True)
        self._forgeable = re.compile("|".join(re.escape(token) for token in added)) if added else None

    def __call__(self, text: str) -> str:
        if self._forgeable is None:
            return text
        return self._forgeable.sub(lambda match: c.SANITIZE_OPEN + match.group(0)[1:-1] + c.SANITIZE_CLOSE, text)
