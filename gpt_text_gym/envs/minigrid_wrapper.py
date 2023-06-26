import minigrid
import gymnasium as gym

from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from gpt_text_gym.envs.base_wrapper import BaseTextEnvWrapper
from gpt_text_gym.gpt import Message
from gpt_text_gym.gpt.utils import remove_leading_whitespace
from typing import Any


class MinigridTextEnvWrapper(BaseTextEnvWrapper):
    def __init__(self):
        self._prompt = ""

    def generate_message(self, env: gym.Env, obs: Any) -> Message:
        """Describe the environment and observation as a message"""
        return Message(role="user", content=self.make_message(env, obs))

    def generate_action(self, env: gym.Env, message: Message) -> Any:
        """Generate an action from a message"""
        return Actions[message.content]

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        self._prompt = prompt

    def make_message(self, env, obs):
        return make_message(env, obs, self.prompt)


def make_minigrid_description() -> str:
    colors = list(COLOR_TO_IDX.keys())
    objects = list(OBJECT_TO_IDX.keys())

    return remove_leading_whitespace(
        f"""
        You are an agent in a gridworld.
        The environment is a gridworld with a 2D view from above. 
        It contains a single agent and a number of objects.

        The possible colors are:
        {", ".join(colors)}

        The possible objects are:
        {", ".join(objects)}

        The possible actions are:
        {", ".join([action.name for action in Actions])}
        """,
        8,
    )


def make_env_description(env, obs) -> str:
    return remove_leading_whitespace(
        """
        The environment state is represented by a grid of size {2 * env.width}x{env.height}.
        Eacg grid cell is described by a 2-character string, the first one for
        the object and the second one for the color.
        An empty grid cell is represented by the string "  ".

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "locked_door": "L",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map of colors to short string
        COLOR_TO_STR = {
            "red": "R",
            "green": "G",
            "blue": "B",
            "purple": "P",
            "yellow": "Y",
            "grey": "G",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
    """
        + f"""
        The environment state is:
        {str(env.unwrapped)}

        The mission is: 
        {obs["mission"]}        
    """
    )


def make_rules() -> str:
    return remove_leading_whitespace(
        """
        The rules of the environment are:
        1. You can pick up an object if you are standing on it.
        2. You can drop an object if you are holding it.
        3. You can toggle an object if it is in front of you.
        4. You can move forward, turn left, or turn right.
        5. You can only pick up an object if you are not holding anything.
        6. When you drop an object, it will be placed on the grid cell you are standing on.
        7. You cannot walk through walls. If you try, you will stay in the same place.
        8. You cannot walk through locked doors. If you try, you will stay in the same place.
        9. You can unlock a locked door with the correct key.
        10. You cannot walk over objects. If you try, you will stay in the same place.
    """
    )


# TODO: Make prompt easily configurable
def make_prompt():
    return remove_leading_whitespace(
        """
        Think about it carefully. What overall plan should you follow to complete the mission?
        The steps of the plan should be simple and easy to follow.
        You should not describe the plan in too much detail.
        """
    )


def make_message(env, obs, prompt=""):
    if prompt == "":
        prompt = make_prompt()
    return f"""
        {make_minigrid_description()}
        {make_env_description(env, obs)}
        {make_rules()}
        {prompt}
    """


if __name__ == "__main__":
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode=None)
    obs, _ = env.reset()

    prompt = make_message(env, obs)
    print(prompt)
