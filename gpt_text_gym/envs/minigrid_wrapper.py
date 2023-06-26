import minigrid
import gymnasium as gym

from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from gpt_text_gym.envs.base_wrapper import BaseTextEnvWrapper
from gpt_text_gym.gpt import Message
from typing import Any


class MinigridTextEnvWrapper(BaseTextEnvWrapper):
    def generate_message(env: gym.Env, obs: Any) -> Message:
        """Describe the environment and observation as a message"""
        return Message(role="user", content=make_prompt(env, obs))

    def generate_action(env: gym.Env, message: Message) -> Any:
        """Generate an action from a message"""
        return Actions[message.content]


def make_minigrid_description() -> str:
    colors = list(COLOR_TO_IDX.keys())
    objects = list(OBJECT_TO_IDX.keys())

    return (
        f"""
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
"""
    )


def make_env_description(env, obs) -> str:
    return f"""
The environment state is:
{str(env.unwrapped)}

The mission is: 
{obs["mission"]}        
"""


def make_rules() -> str:
    return """
The rules of the environment are:
1. You can pick up an object if you are standing on it.
2. You can drop an object if you are holding it.
3. You can toggle an object if it is in front of you.
4. You can move forward, turn left, or turn right.
5. You can only pick up an object if you are not holding anything.
6. When you drop an object, it will be placed on the grid cell you are standing on.
"""


def make_prompt(env, obs):
    return f"""
You are an agent in a gridworld.

{make_minigrid_description()}
{make_env_description(env, obs)}

To solve the environment, you first decide to break down the 
high-level mission into a sequence of intermediate goal states. 
Thinking step by step, the goal states are: 
    """


if __name__ == "__main__":
    env = gym.make("MiniGrid-BlockedUnlockPickup-v0", render_mode=None)
    obs, _ = env.reset()

    prompt = make_prompt(env, obs)
    print(prompt)
