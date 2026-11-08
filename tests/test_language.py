import pytest

from rein.datagen.language import matches_language
from rein.schema import Message


def _conversation(*texts: str) -> tuple[Message, ...]:
    return tuple(Message(role="user" if i % 2 == 0 else "assistant", text=text) for i, text in enumerate(texts))


@pytest.mark.parametrize(
    ("code", "texts"),
    [
        ("pt-BR", ("oi, tudo bem? me manda uma foto aí", "claro, já vou mandar daqui a pouco")),
        ("en", ("hey, can you send me a photo of the menu?", "sure, give me a second")),
        ("ja", ("こんにちは、写真を送ってもらえますか。", "はい、少々お待ちください。")),
        ("pt-BR+en", ("mano, that was insane, você viu o jogo ontem?", "sim! the ending was crazy demais")),
    ],
)
def test_requested_language_is_detected(code: str, texts: tuple[str, ...]) -> None:
    assert matches_language(_conversation(*texts), code)


def test_wrong_language_is_rejected() -> None:
    conversation = _conversation("oi, tudo bem? me manda uma foto aí", "claro, já vou mandar daqui a pouco")
    assert not matches_language(conversation, "hi")
    assert not matches_language(conversation, "en")
