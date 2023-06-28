from gpt_text_gym.envs.minigrid.minigrid_wrapper import Grid


def test_simple_grid():
    import minigrid  # noqa
    import gymnasium as gym

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=None)
    env.reset()
    env_str = str(env.unwrapped)
    env_str = "\n".join([l.strip() for l in env_str.split("\n")])
    grid = Grid.from_string(env_str)
    assert grid.rows == 5
    assert grid.cols == 5
    assert grid.cells[0][0] == "WG"
    env_str_pred = str(grid)
    assert env_str == env_str_pred
