from dataclasses import dataclass, field

from rein.env import load_env, optional, require
from rein.teacher import constants as c


@dataclass(frozen=True)
class TeacherSettings:
    base_url: str
    proxy_key: str | None = field(default=None, repr=False)
    proxy_secret: str | None = field(default=None, repr=False)
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.startswith(c.REQUIRED_URL_SCHEME):
            raise ValueError(f"teacher URL must use {c.REQUIRED_URL_SCHEME}")
        if bool(self.proxy_key) != bool(self.proxy_secret):
            raise ValueError(f"set both {c.ENV_PROXY_KEY} and {c.ENV_PROXY_SECRET}, or neither")

    @classmethod
    def from_env(cls, values: dict[str, str] | None = None) -> "TeacherSettings":
        env = values if values is not None else load_env()
        return cls(
            base_url=require(env, c.ENV_TEACHER_URL).rstrip("/"),
            proxy_key=optional(env, c.ENV_PROXY_KEY),
            proxy_secret=optional(env, c.ENV_PROXY_SECRET),
            model=optional(env, c.ENV_TEACHER_MODEL),
        )

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.proxy_key and self.proxy_secret:
            headers[c.PROXY_KEY_HEADER] = self.proxy_key
            headers[c.PROXY_SECRET_HEADER] = self.proxy_secret
        return headers
