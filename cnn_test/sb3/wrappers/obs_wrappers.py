from typing import Any, Dict, Tuple

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig

P_NUM = 2


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                13,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation


def get_obs_space_spec(env_cfg: EnvConfig) -> gym.spaces.Dict:
    x = env_cfg.map_size
    y = env_cfg.map_size

    return gym.spaces.Dict(
        {
            "robot": spaces.MultiBinary((1, P_NUM, x, y)),
            # 0 - light, 1 - heavy
            "robot_type": spaces.MultiBinary((1, P_NUM, x, y)),
            "factory": spaces.MultiBinary((1, P_NUM, x, y)),
            "robot_power": spaces.Box(0.0, 1.0, shape=(1, P_NUM, x, y)),
            "robot_cargo_ice": spaces.Box(0.0, 1.0, shape=(1, P_NUM, x, y)),
            "robot_cargo_ore": spaces.Box(0.0, 1.0, shape=(1, P_NUM, x, y)),
            "robot_cargo_water": spaces.Box(0.0, 1.0, shape=(1, P_NUM, x, y)),
            "robot_cargo_metal": spaces.Box(0.0, 1.0, shape=(1, P_NUM, x, y)),
        }
    )


def get_empty_obs_spec(env_cfg: EnvConfig) -> Dict[str, np.ndarray]:
    _empty_obs = {}
    for key, spec in get_obs_space_spec(env_cfg).spaces.items():
        if isinstance(spec, gym.spaces.MultiBinary) or isinstance(
            spec, gym.spaces.MultiDiscrete
        ):
            _empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
        elif isinstance(spec, gym.spaces.Box):
            _empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
        else:
            raise NotImplementedError(
                f"{type(spec)} is not an accepted observation space."
            )
    return _empty_obs


class SimpleMutiDimObsSpace(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = get_obs_space_spec(env.state.env_cfg)

    def observation(self, obs):
        return SimpleMutiDimObsSpace.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: EnvConfig) -> Dict[str, npt.NDArray]:

        observation = get_empty_obs_spec(env_cfg)
        # ice_map = obs["board"]["ice"]
        # ice_tile_locations = np.argwhere(ice_map == 1)
        if "teams" in obs.keys():
            players = obs["teams"].keys()
        else:
            players = obs.keys()

        # TODO: FIX this for eval
        shared_obs = obs["player_0"]

        for agent in players:

            p_id = int(agent.split("_")[1])

            factories = shared_obs["factories"][agent]
            for k in factories.keys():
                x, y = factories[k]["pos"]
                observation["factory"][0, p_id, x, y] = 1

            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]
                x, y = unit["pos"]

                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                # store cargo+power values scaled to [0, 1]
                # print(unit["power"] / battery_cap)
                # power = unit["power"] / battery_cap
                # print(type(observation["robot_power"][0, p_id, x, y]))

                # observation["robot_power"][0, p_id, x, y] = float(
                #     unit["power"] / battery_cap,
                # )

                observation["robot_cargo_ice"][0, p_id, x, y] = unit["cargo"]["ice"] / cargo_space
                observation["robot_cargo_ore"][0, p_id, x, y] = (
                    unit["cargo"]["ore"] / cargo_space
                )
                observation["robot_cargo_metal"][0, p_id, x, y] = (
                    unit["cargo"]["metal"] / cargo_space
                )
                observation["robot_cargo_water"][0, p_id, x, y] = (
                    unit["cargo"]["water"] / cargo_space
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                observation["robot_type"][0, p_id, x, y] = unit_type

                break

        return observation
