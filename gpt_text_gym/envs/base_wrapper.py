import abc
import gymnasium as gym

from gpt_text_gym.gpt.chat_completer import Message
from typing import Any


class BaseTextEnvWrapper(abc.ABC):
    @abc.abstractmethod
    def generate_message(env: gym.Env, obs: Any) -> Message:
        """Describe the environment and observation as a message"""
        pass

    @abc.abstractmethod
    def generate_action(env: gym.Env, message: Message) -> Any:
        """Generate an action from a message"""
        pass
