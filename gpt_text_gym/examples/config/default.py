from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    """Returns the default config"""
    config = config_dict.ConfigDict()

    config.env_id: str = "MiniGrid-BlockedUnlockPickup-v0"
    config.gpt = config_dict.ConfigDict()
    config.gpt.model: str = "gpt-4"

    return config
