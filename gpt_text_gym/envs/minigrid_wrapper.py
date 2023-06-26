import minigrid
import gymnasium as gym

from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from gpt_text_gym.envs.base_wrapper import BaseTextEnvWrapper
from gpt_text_gym.gpt import Message
from gpt_text_gym.gpt.utils import remove_leading_whitespace
from typing import Any


class MinigridTextEnvWrapper(BaseTextEnvWrapper):
    def generate_message(self, env: gym.Env, obs: Any) -> Message:
        """Describe the environment and observation as a message"""
        return Message(role="user", content=make_message(env, obs))

    def generate_action(self, env: gym.Env, message: Message) -> Any:
        """Generate an action from a message"""
        return Actions[message.content]


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
        """
        + """
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.

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

        # Enumeration of possible actions
        class Actions(IntEnum):
            # Turn left, turn right, move forward
            left = 0
            right = 1
            forward = 2
            # Pick up an object
            pickup = 3
            # Drop an object
            drop = 4
            # Toggle/activate an object
            toggle = 5

            # Done completing task
            done = 6
        """,
        8,
    )


def make_env_description(env, obs) -> str:
    return remove_leading_whitespace(
        f"""
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


def make_prompt():
    return remove_leading_whitespace(
        """
        1. What is the mission?
        2. Can you walk through walls?
        3. Are you in the same room as the goal object?
        4. How can you get to the goal object?
        5. How do you get to the goal object if you are blocked by a locked door and walls?
    """
    )


def make_message(env, obs):
    return f"""
        {make_minigrid_description()}
        {make_env_description(env, obs)}
        {make_rules()}
        {make_prompt()}
    """


if __name__ == "__main__":
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode=None)
    obs, _ = env.reset()

    prompt = make_message(env, obs)
    print(prompt)
