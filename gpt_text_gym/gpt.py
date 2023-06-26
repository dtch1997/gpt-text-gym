""" Interface to GPT model."""

import openai
import dotenv

from gpt_text_gym import ROOT_DIR
from typing import List, NewType, Dict

Message = NewType("Message", Dict[str, str])


def pretty_print_message(message: Message):
    print(f"{message['role']}: {message['content']}")


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

    def generate_chat_completion(self, messages: List[Message], **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        choice = response["choices"][0]
        msg: Message = choice["message"]
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
    reply = chatbot.generate_chat_completion(messages)
    for message in messages:
        pretty_print_message(message)
    pretty_print_message(reply)
