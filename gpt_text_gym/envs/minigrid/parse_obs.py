""" Utilities to parse the observation """

import minigrid
import numpy as np
import gymnasium as gym

from dataclasses import dataclass
from typing import List
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR


@dataclass
class Observation:
    mission: str
    current_objective: str
    environment: List[str]
    inventory: str


def is_current_objective_achieved(obs: Observation) -> bool:
    pass


def get_objects(img_obs: np.ndarray):
    """Parse Minigrid observation"""
    rows, cols, _ = img_obs.shape

    objects = []
    for row in range(rows):
        for col in range(cols):
            obj_idx, color_idx, state_idx = img_obs[row, col]
            type = IDX_TO_OBJECT[obj_idx]
            color = IDX_TO_COLOR[color_idx]

            color += " "
            # Define color
            if type == "unseen":
                color = ""

            # Define state
            if type == "door":
                state = {0: "open", 1: "closed", 2: "locked"}[state_idx]
                state += " "
            else:
                state = ""
            objects.append(f"{state}{color}{type} at ({row}, {col})")
    return objects


def get_inventory(env):
    carrying = env.unwrapped.carrying
    if carrying is None:
        return "nothing"
    else:
        return f"{carrying.color} {carrying.type}"


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )
    env = minigrid.wrappers.FullyObsWrapper(env)

    obs, _ = env.reset()

    # Natural language description of environment
    objects = get_objects(obs["image"])
    print("Objects in environment:")
    for object in objects:
        print(object)

    print("Inventory:")
    print(get_inventory(env))

    env.render()
    input("Press enter to continue...")
