"""Utility functions."""
from __future__ import annotations  # PEP 604 backport

import itertools
from collections.abc import Callable
import numpy as np

# Global variable to store cached estimates for model configurations.
# This should be used with caution and only when the hyperparameter grid is consistent:
# e.g., inside a loop to find a valid configuration.
cached_estimates: list[tuple[dict[str, float], float]] | None = None


def search_model_config(
    N_opt: float,
    hyperparam_grid: dict[str, list[float]],
    size_estimator: Callable[..., float],
    use_cache: bool = False,
) -> tuple[dict[str, float], float]:
    """
    Finds the model configuration that is closest to a given target number of parameters,
    based on the provided hyperparameter grid and size estimator.

    > **Example Usage**:
    > ```python
    > from chinchilla._utils import search_model_config
    > def my_size_estimator_function(hidden_size, num_layers):
    >     # ... Write your code to estimate `N`
    >
    > model_search_config = {
    >     hyperparam_grid={
    >         'hidden_size': [1024, 2048, 4096],
    >         'num_layers': [1, 2, 4, 8, 16, 32]
    >     },
    >     size_estimator=my_size_estimator_function
    > }
    > model_config, estimated_size = search_model_config(1e9, **model_search_config)
    > ```

    Args:
        N_opt (float): The target number of parameters for the model configuration.
        hyperparam_grid (dict[str, list[float]]): A dictionary where keys are hyperparameter names and values are lists of possible values.
        size_estimator (Callable[..., float]): A callable that takes a model configuration as keyword arguments and returns the estimated size.
        use_cache (bool, optional): A boolean flag indicating whether to use cached estimates. Default is False.

    Returns:
        A tuple containing the closest model configuration and its estimated size.

    Note:
        Although very efficient, you should set `use_cache` to True only when `hyperparam_grid` is guaranteed to be
        consistent; thus, it is disabled by default except for Simulator (x16 faster).
    """
    global cached_estimates
    if not use_cache or cached_estimates is None:
        # Pre-compute and sort estimates if not already done
        estimates = []
        for _values in itertools.product(*hyperparam_grid.values()):
            config = dict(zip(hyperparam_grid.keys(), _values))
            N = size_estimator(**config)
            if N and np.isfinite(N):
                estimates.append((config, N))
        estimates.sort(key=lambda x: x[1])  # Sort by estimated size
        cached_estimates = estimates

    closest = cached_estimates[np.abs(np.array([e[1] for e in cached_estimates]) - N_opt).argmin()]
    return closest


def is_between(value: float | np.ndarray, bounds: tuple[float, float] | np.ndarray) -> bool | np.ndarray:
    """
    Checks if a value is within the given inclusive bounds.

    Args:
        value (float | np.ndarray): The value to check.
        bounds (tuple[float, float] | np.ndarray): A tuple containing the lower and upper bounds.

    Returns:
        bool | np.ndarray: NumPy array: A boolean or an NumPy array of booleans indicating whether the value is between the bounds.
    """
    lower, upper = bounds
    return lower <= value <= upper
