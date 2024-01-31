"""Pydantic classes for validating parameters and configurations for `Chinchilla` and `Simulator`."""

from __future__ import annotations  # PEP 604 backport

from pydantic import BaseModel, Field, model_validator, field_validator

from typing import Optional, Callable, Iterable, Iterator, Tuple, Dict
# We use capitalized types here because built-in generic types (PEP 585) supported only from python 3.9

VALID_KEYS_IN_LOWER = "e", "a", "b", "alpha", "beta"


class ParamGrid(BaseModel):
    """
    Validates a grid of initialization for scaling law (/loss predictor) parameters.

    Attributes:
        E or e: Tuple of floats representing initial values for the E parameter or its log form.
        A or a: Tuple of floats representing initial values for the A parameter or its log form.
        B or b: Tuple of floats representing initial values for the B parameter or its log form.
        alpha: Tuple of floats representing initial values for the alpha parameter.
        beta: Tuple of floats representing initial values for the beta parameter.
    """

    E: Optional[Tuple[float, ...]] = None
    A: Optional[Tuple[float, ...]] = None
    B: Optional[Tuple[float, ...]] = None
    alpha: Tuple[float, ...]
    beta: Tuple[float, ...]
    e: Optional[Tuple[float, ...]] = None
    a: Optional[Tuple[float, ...]] = None
    b: Optional[Tuple[float, ...]] = None

    @model_validator(mode="before")
    def check_keys(cls, values):
        """Validates that the parameter grid contains the correct keys."""
        keys = [k.lower() for k in values.keys()]
        if len(set(keys)) != 5:
            raise ValueError("`param_grid` must be a dict of 5 keys")
        if len(set(keys)) != len(keys):
            raise ValueError("There are effectively duplicate keys when lowercased.")

        invalid_keys = [
            k
            for k in values.keys()
            if (len(k) == 1 and k.lower() not in VALID_KEYS_IN_LOWER) or (len(k) > 1 and k not in VALID_KEYS_IN_LOWER)
        ]
        missing_keys = [p for p in VALID_KEYS_IN_LOWER if p not in [k.lower() for k in keys]]
        missing_keys = [f"{p.upper()}/{p}" if len(p) == 1 else p for p in missing_keys]
        # duplicate_keys = []
        if invalid_keys + missing_keys:
            raise ValueError(
                "Invalid set of keys in `param_grid`:\n"
                + (f"- Invalid key(s): {', '.join(invalid_keys)}\n" if invalid_keys else "")
                + (f"- Missing key(s): {', '.join(missing_keys)}\n" if missing_keys else "")
            )
        return values


class SeedRanges(BaseModel):
    """
    Validates the regime of seed models.

    Attributes:
        C: Tuple of two floats representing the range for the `C` parameter.
        N_to_D: Tuple of two floats representing the range for the `N_to_D` parameter.
    """

    C: Tuple[float, float]
    N_to_D: Tuple[float, float]

    @field_validator("C")
    def check_C_values(cls, values):
        """Validates the values for the 'C' range."""
        if any(c < 0 for c in values):
            raise ValueError("Negative value(s) in 'C': Consider changing data type.")
        if not values[0] < values[1]:
            raise ValueError("'C' must have the first value less than the second value.")
        return values


class ModelSearchConfig(BaseModel):
    """
    Validates keyword arguments into `_utils.search_model_config`.

    Attributes:
        size_estimator: A callable that estimates the size of the model.
        hyperparam_grid: A dictionary representing the hyperparameter grid.
    """

    size_estimator: Callable[[], float]
    hyperparam_grid: Dict[str, Tuple[float, ...]]

    @field_validator("size_estimator")
    def size_estimator_must_be_callable(cls, v):
        """Validates that the size estimator is a callable."""
        if not callable(v):
            raise TypeError("size_estimator must be a callable (function or class method).")
        return v

    @field_validator("hyperparam_grid")
    def hyperparam_grid_must_be_dict(cls, v):
        """Validates that the hyperparameter grid is a dictionary."""
        if not isinstance(v, dict):
            raise TypeError("hyperparam_grid must be a dictionary.")
        return v


class SimulationArgs(BaseModel):
    """
    Validates arguments for running simulations.

    Attributes:
        num_seeding_steps: The number of seeding steps.
        num_scaling_steps: The number of scaling steps.
        target_params: A dictionary representing hypothetical scaling law parameters.
        noise_generator: An optional iterable of floats representing the additional loss randomly caused by imperfect training.
        scaling_factor: An optional float representing the scaling factor in FLOPs.
    """

    num_seeding_steps: int = Field(gt=0)
    num_scaling_steps: int = Field(ge=0)
    target_params: Dict[str, float] = (
        {"E": 1.69337368, "A": 406.401018, "B": 410.722827, "alpha": 0.33917084, "beta": 0.2849083},
    )
    noise_generator: Optional[Iterable[float]] = None
    scaling_factor: Optional[float] = None

    @model_validator(mode="before")
    def check_seeding_steps(cls, values):
        """Validates the number of seeding and scaling steps along with the scaling factor."""
        num_seeding_steps = values.get("num_seeding_steps")
        num_scaling_steps = values.get("num_scaling_steps")
        scaling_factor = values.get("scaling_factor")  # Updated with self.scaling_factor if None
        if num_seeding_steps < 3 and num_scaling_steps:
            raise ValueError("Too few seed models to estimate the scaling law.")
        if num_scaling_steps and scaling_factor is None:
            raise ValueError(
                "Chinchilla object must be assigned a positive float to `scaling_factor` when simulating the scaling process. "
                "You may specify it in `cc = Chinchilla(...)` or `cc.simulate(...)`"
            )
        return values

    @field_validator("target_params")
    def check_target_params(cls, v):
        """Validates that the target parameters are a dictionary with float or int values."""
        if not isinstance(v, dict) or not all(isinstance(p, (float, int)) for p in v.values()):
            raise ValueError("target_params must be a dictionary with float or int values.")
        return v

    @field_validator("noise_generator")
    def check_noise_generator(cls, v):
        """Validates the noise generator, ensuring it is an iterator or None."""
        if v is not None and not isinstance(v, Iterator):
            raise ValueError("noise_generator must be an iterator or None")
        return v
