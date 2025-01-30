"""
This module defines the `Environment` class and related components for creating
and managing grid-based environments using the MiniGrid framework.

Classes:
    BaseSettings: A dataclass that holds configuration settings for the environment.
    Environment: A class that initializes the environment with default settings.

Attributes:
    __all__: A list of public objects of this module, as interpreted by `import *`.

Usage:
    - Define environment settings using the `BaseSettings` dataclass.
    - Create an `Environment` instance with the default settings.
    - Use the `map_generator` method to generate environments with specific configurations.
"""

import dataclasses as dc
from typing import Type

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

__all__ = ["Environment", "Goal", "BaseSettings", "Wall"]


@dc.dataclass
class BaseSettings:
    """
    BaseSettings class defines the configuration settings for the environment.

    Attributes:
        grid_size (int | None): Size of the grid.
        width (int | None): Width of the grid.
        height (int | None): Height of the grid.
        max_steps (int): Maximum number of steps. Default is 100.
        see_through_walls (bool): Whether the agent can see through walls. Default is False.
        agent_view_size (int): Size of the agent's view. Default is 7.
        render_mode (str | None): Rendering mode.
        screen_size (int | None): Size of the screen. Default is 640.
        highlight (bool): Whether to highlight the agent. Default is True.
        agent_pov (bool): Whether to use the agent's point of view. Default is False.

    Methods:
        __post_init__(): Validates the fields of the class after initialization.
    """

    grid_size: int | None = None  # size of the grid
    width: int | None = None  # width of the grid
    height: int | None = None  # height of the grid
    max_steps: int = 100  # maximum number of steps
    see_through_walls: bool = False  # whether the agent can see through
    agent_view_size: int = 7  # size of the agent's view
    render_mode: str | None = None  # rendering mode
    screen_size: int | None = 640  # size of the screen
    highlight: bool = True  # whether to highlight the agent
    agent_pov: bool = False  # whether to use the agent's point of view

    def __post_init__(self):
        allowed = {f.name for f in dc.fields(BaseSettings)}
        for field in self.__dict__:
            if field not in allowed:
                raise TypeError(f"Unexpected field: {field}")


class Environment:
    """
    A class to represent an environment for spatial navigation.

    Attributes:
    ----------
    defaults : Type[BaseSettings]
        Default settings for the environment.

    Methods:
    -------
    map_generator(name: str)
        Generates a map with the given name using the default settings.
    """

    def __init__(self, defaults: Type[BaseSettings]):
        self.defaults = defaults

    def map_generator(self, name: str):
        """
        Generates a map environment based on the given name.

        Args:
            name (str): The name of the environment to generate.

        Returns:
            _EnvironGen: An instance of the _EnvironGen class initialized with the given name and default settings.
        """
        return _EnvironGen(name, self.defaults)


class _EnvironGen:  # pylint: disable=too-few-public-methods
    def __init__(self, name: str, defaults: Type[BaseSettings]):
        self.mission_space = MissionSpace(mission_func=lambda: name)
        self.defaults = defaults

    def __call__(self, func):
        return _create_env(func, self.mission_space, self.defaults)


def _create_env(func, mission_space, defaults):

    class _BaseEnvironment(MiniGridEnv):
        def __init__(self, **kwds):
            super().__init__(mission_space, **defaults(**kwds).__dict__)

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            return func(self, self.grid, width, height)

    return _BaseEnvironment
