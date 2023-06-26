import minigrid  # noqa
import gymnasium as gym

from gpt_text_gym.envs.minigrid_wrapper import MinigridTextEnvWrapper
from gpt_text_gym.gpt import GPTChatCompleter, Message, default_system_message

from absl import app
from absl import flags
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config", None)
flags.mark_flag_as_required("config")


def get_config():
    return _CONFIG.value


def main(argv):
    config = get_config()
    env = gym.make(config.env_id, render_mode=None)
    wrapper = MinigridTextEnvWrapper()

    obs, _ = env.reset()
    prompt = wrapper.generate_message(env, obs)
    if config.help:
        # Print the prompt and exit
        print(prompt)
        return

    chatbot = GPTChatCompleter(
        model=config.gpt.model,
    )
    chatbot.add_message(default_system_message())
    chatbot.add_message(prompt)
    reply = chatbot.generate_chat_completion()

    print(prompt)
    print(reply)


if __name__ == "__main__":
    app.run(main)
