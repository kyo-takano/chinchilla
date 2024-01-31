from __future__ import annotations  # PEP 604 backport

import random
from typing import Callable, Iterator, Any

import numpy as np

from rich.progress import track

from ._logger import get_logger
from ._validator import SimulationArgs
from .core import Chinchilla
from .database import Database


class Simulator(Chinchilla):
    """
    Simulates the scaling law estimation with `Chinchilla`, allowing you to understand its behaviors.

    Inheriting and extending the `Chinchilla` class with the capacity to simulate seeding and scaling in a hypothetical task,
    `Simulator` models how factors like `Chinchilla` configuration, number of seeds, scaling factor, the noisiness of losses, etc.
    would confound to affect the stability and the performance of scaling law estimation.

    Attributes:
        cc (Chinchilla): The Chinchilla instance with preset configuration attributes like `param_grid`, `seed_ranges`, `loss_fn`, etc.
        logger (Logger): Logger instance for simulation activities.
        database (Database): Database instance to record simulation results.

    Methods:
        __init__: Initializes the Simulator with a Chinchilla instance.
        __getattr__: Delegates attribute access to the Chinchilla instance.
        __call__: Executes the simulation with given parameters.
        _pseudo_training_run: Performs a pseudo-training run and records results.
    """

    def __init__(self, cc: Chinchilla):
        """
        Initialize the Simulator with a Chinchilla instance.

        Args:
            cc (Chinchilla): An instance of Chinchilla to be used for simulation.
        """
        self.cc = cc
        self.log_level = self.cc.logger.getEffectiveLevel()
        self.logger = get_logger(self.log_level, name="chinchilla.simulator")
        self._create_shortcuts()

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the Chinchilla instance.

        Args:
            name: The name of the attribute to access.

        Returns:
            Any: The value of the attribute with the specified name from the Chinchilla instance.
        """
        return getattr(self.cc, name)

    def __call__(
        self,
        num_seeding_steps: int = 100,
        num_scaling_steps: int = 1,
        scaling_factor: float | None = None,
        target_params: dict[str, float] = {
            "E": 1.69337368,
            "A": 406.401018,
            "B": 410.722827,
            "alpha": 0.33917084,
            "beta": 0.2849083,
        },
        noise_generator: Iterator | tuple[Callable, tuple[float, ...]] | None = (
            random.expovariate(10) for _ in iter(int, 1)
        ),
    ) -> None:
        """
        Simulate the compute-scaling on a hypothetical deep learning task with some noise expectable from reality.

        Args:
            num_seeding_steps (int): The number of seeding steps to simulate.
            num_scaling_steps (int): The number of scaling steps to simulate.
            scaling_factor (float | None, optional): The scaling factor to be used in the simulation.
            target_params (dict[str, float]): A dictionary of target parameters for the simulation.
            noise_generator (Iterator | tuple[Callable, tuple[float, ...]] | None, optional): A callable or iterator that generates noise to be added to the loss.
                Defaults to `(random.expovariate(10) for _ in iter(int, 1))`, which generates an exponential distribution averaging at $0.100$.

        Raises:
            TypeError: If the provided `noise_generator` is not an iterator or a tuple with a callable and its arguments.
        """
        # Preprocessing: if `noise_generator` is passed as a pair of method and args, convert into an infinite generator
        # e.g., `(random.expovariate, (10,))`
        if isinstance(noise_generator, (tuple, list)):
            func, args = noise_generator
            if not (callable(func) and isinstance(args, (tuple, list))):
                raise TypeError(
                    "When passing a callable and arguments as `noise_generator`, it must be executable as `func(*args)`"
                )
            noise_generator = (func(*args) for _ in iter(int, 1))

        # Validation
        SimulationArgs(
            num_seeding_steps=num_seeding_steps,
            num_scaling_steps=num_scaling_steps,
            scaling_factor=scaling_factor or self.scaling_factor,
            target_params=target_params,
            noise_generator=noise_generator,
        )
        self.num_seeding_steps = num_seeding_steps
        self.num_scaling_steps = num_scaling_steps
        self.scaling_factor = scaling_factor
        self.target_params = target_params
        self.noise_generator = noise_generator

        # Reset with an in-memory database *not* saving appended data.
        self.database = Database(log_level=self.log_level)
        self.logger.info("[b]Starting a simulation[/]")
        # Main: simulate seeding and scaling
        for _ in track(
            range(self.num_seeding_steps),
            description="  [SIMULATION] Seed training runs ",
            disable=self.log_level <= 10 or 30 < self.log_level,  # logging.WARNING
        ):
            self._pseudo_training_run()

        self.logger.debug("Data types:\n" + self.database.df.dtypes.T.to_string())

        for _ in range(self.num_scaling_steps):  # Logger will show the progress
            self._pseudo_training_run()

        if not num_scaling_steps:
            # Plot the distribution of seed models with the estimated loss gradient
            self.plot(simulation=True)

    def _pseudo_training_run(self) -> None:
        """Perform a pseudo training run and record the results in the database."""
        (N, D), _ = self.step(simulation=True)

        # Get a *hypothetical* lowest loss you can achieve with N and D
        pseudo_loss = sum(
            [
                self.target_params["E"],
                np.exp(np.log(self.target_params["A"]) - self.target_params["alpha"] * np.log(N)),
                np.exp(np.log(self.target_params["B"]) - self.target_params["beta"] * np.log(D)),
            ]
        )
        # Add a random noise (>0.0) attributed to inefficiency of training setup
        if self.noise_generator:
            pseudo_loss += next(self.noise_generator)

        self.database.append(N=N, D=D, C=6 * N * D, loss=pseudo_loss)  # type: ignore
