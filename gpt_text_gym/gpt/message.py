from dataclasses import dataclass
from typing import NewType, Dict

RawMessage = NewType("Message", Dict[str, str])


@dataclass
class Message:
    role: str
    content: str

    @staticmethod
    def from_dict(raw_message: RawMessage) -> "Message":
        return Message(role=raw_message["role"], content=raw_message["content"])

    def to_dict(self) -> RawMessage:
        return RawMessage({"role": self.role, "content": self.content})

    def __str__(self):
        return f"{self.role}: {self.content}"


def default_system_message():
    return Message(role="system", content="You are a helpful assistant.")
