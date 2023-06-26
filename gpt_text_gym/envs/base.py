import abc
from gpt_text_gym.gpt import Message


class BaseTextEnvWrapper(abc.ABC):
    @abc.abstractmethod
    def generate_message(env, obs) -> Message:
        """Describe the environment and observation as a message"""
        pass

    @abc.abstractmethod
    def generate_action(env, message: Message):
        """Generate an action from a message"""
        pass
