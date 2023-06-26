import minigrid  # noqa
import gymnasium as gym

from gpt_text_gym.envs.minigrid_wrapper import MinigridTextEnvWrapper
from gpt_text_gym.gpt import GPTChatCompleter, Message, default_system_message

if __name__ == "__main__":
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode=None)
    wrapper = MinigridTextEnvWrapper()

    obs, _ = env.reset()
    prompt = wrapper.generate_message(env, obs)

    chatbot = GPTChatCompleter()
    chatbot.add_message(default_system_message())
    chatbot.add_message(prompt)
    reply = chatbot.generate_chat_completion()

    print(prompt)
    print(reply)
