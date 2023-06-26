""" Interface to GPT model."""

import openai
import dotenv

from dataclasses import dataclass
from gpt_text_gym import ROOT_DIR
from typing import List, NewType, Dict, Optional

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


def openai_chat_completion_create(
    model: str,
    messages: List[RawMessage],
    n: int,
    temperature: float,
    max_tokens: Optional[int],
    **kwargs,
):
    """Wrapper around OpenAI's ChatCompletion.create method."""
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


class GPTChatCompleter:
    def __init__(self):
        openai.api_key = dotenv.get_key(ROOT_DIR / ".env", "API_KEY")
        self.chat_history: List[Message] = []
        self.model = "gpt-3.5-turbo"
        self.temperature = 1.0
        self.max_tokens = None
        self.n = 1

    def clear(self):
        self.chat_history = []

    def generate_chat_completion(self, **kwargs):
        messages = [message.to_dict() for message in self.chat_history]
        response = openai_chat_completion_create(
            model=self.model,
            messages=messages,
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        choice = response["choices"][0]
        msg: Message = Message.from_dict(choice["message"])
        return msg

    def add_message(self, message: Message):
        self.chat_history.append(message)


if __name__ == "__main__":

    chatbot = GPTChatCompleter()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Translate the following English text to French: 'Hello, how are you?'",
        },
    ]
    messages = [Message.from_dict(message) for message in messages]
    for message in messages:
        print(message)
        chatbot.add_message(message)
    reply = chatbot.generate_chat_completion(messages)
    print(reply)
