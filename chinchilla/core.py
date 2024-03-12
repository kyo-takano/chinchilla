from __future__ import annotations  # PEP 604 backport

import itertools
import multiprocessing
import os
import random
import textwrap
from collections.abc import Callable

import numpy as np
from attrdictx import AttrDict
from ordinal import ordinal
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy import optimize as sciop  # import fmin_l_bfgs_b  # , minimize

from ._logger import get_logger
from ._metrics import asymmetric_mae
from ._utils import is_between, search_model_config
from ._validator import ModelSearchConfig, ParamGrid, SeedRanges
from .database import Database  # Used for initializing database connections
from .visualizer import Visualizer  # Required for setting up visualization properties

# 128bit/96bit: more precise than 64bit  at the cost of approx. 2x more time
DTYPE = np.longdouble
# 64bit: yields a *slightly different*, plausibly less precise result;
# Recommendable exclusively for agile testing
# DTYPE = np.double

# Clip values by lower precision for stability with `loss_fn` and `weight_fn`.
FLOAT_TINY = np.finfo(np.single).tiny
FLOAT_LOGMAX = np.log(np.finfo(np.single).max)


class Chinchilla:
    """
    Estimates the scaling law for a deep learning task.
    Provides functionalities to:

    1. Sample models from a specified "seed" regime.
    2. Fit the loss predictor $L(N, D)$.
    3. Suggest an allocation of scaled compute.

    This module includes the `Chinchilla` class, which provides methods for sampling model configurations,
    fitting the parametric loss predictor, suggesting allocations for scaled compute budgets, etc.
    It operates in a numerical precision of **128-bit** by default and integrates with
    [`chinchilla.Database`](#chinchilladatabaseDatabase) and
    [`chinchilla.Visualizer`](#chinchillavisualizerVisualizer)
    for storing and plotting data.
    """

    E: float
    A: float
    B: float
    alpha: float
    beta: float

    def __init__(
        self,
        project_dir: str,
        param_grid: dict[str, np.ndarray | list | tuple],
        seed_ranges: dict[str, np.ndarray | list | tuple],
        model_search_config: dict[str, Callable | dict] | None = None,
        loss_fn: Callable = asymmetric_mae,  # Fits to the floor (\approx. lower bound) of the distribution $L(N, D)$
        weight_fn: Callable | None = None,  # You nay weight loss prediction errors with any input
        num_seeding_steps: int | None = None,
        scaling_factor: float | None = None,
        log_level: int | str = 20,  # logging.INFO
    ) -> None:
        """
        Initializes a Chinchilla instance with the given parameters and sets up the scaling process.

        Args:
            project_dir (str): The directory path for the project where the database and visualizations will be stored.
            param_grid (dict[str, np.ndarray | list | tuple]): A dictionary specifying the grid of parameters to search over.
            seed_ranges (dict[str, np.ndarray | list | tuple]): A dictionary specifying the ranges for seeding the model configurations.
            model_search_config (dict[str, Callable | dict] | None, optional): Configuration for model search. Defaults to None.
            loss_fn (Callable, optional): The loss function to be used for fitting. Defaults to asymmetric_mae.
            weight_fn (Callable | None, optional): A function to weight loss prediction errors. Defaults to None.
            num_seeding_steps (int | None, optional): The number of seeding steps to perform. Defaults to None.
            scaling_factor (float | None, optional): The scaling factor to be used when scaling up compute. Defaults to None.
            log_level (int | str, optional): Specifies the threshold for logging messages.
                A value of 30 suppresses standard messages while any larger values hide all messages entirely. Defaults to 20 (`logging.INFO`).

        Raises:
            ValueError: If `project_dir` does not exist or is not a directory.
            TypeError: If `loss_fn` or `weight_fn` is not callable.
            FileExistsError: If a file with the same name as `project_dir` already exists.
        """
        if not project_dir:
            raise ValueError("Please specify a directory. If it does not exist, it will be created.")

        self.logger = get_logger(log_level, name="chinchilla")

        # input validation
        ParamGrid(**param_grid)
        SeedRanges(**seed_ranges)
        if model_search_config:
            ModelSearchConfig(**model_search_config)
        else:
            self.logger.warning(
                textwrap.dedent(
                    """\
                You did not specify `model_search_config`. In order to find a model configuration, You will need to either:
                1.  [b]Assign `model_search_config` attribute[/] [i]before[/] calling `seed` method.
                2.  Find the model configuration closest to `N` yourself, and optionally call `cc.adjust_D_to_N(N)` when scaling."""
                )
            )
        if not callable(loss_fn):
            raise TypeError("`loss_fn` must be callable")
        if weight_fn and not callable(loss_fn):
            raise TypeError("`weight_fn` must be callable or None")

        # Convert dict to AttrDict for easy access
        seed_ranges = AttrDict(seed_ranges)

        """Initialize configurations"""
        # Seed
        self.seed_ranges = AttrDict(
            # User-specified
            C=[float(c) for c in seed_ranges.C],  # tuple/list of large integers (>2 ** 63) can result in errors
            N_to_D=seed_ranges.N_to_D,
            # Pre-compute the bounds of allocations for the seed models
            N=[
                np.sqrt(seed_ranges.C[0] / (6 * seed_ranges.N_to_D[1])),  # lower bound
                np.sqrt(seed_ranges.C[1] / (6 * seed_ranges.N_to_D[0])),  # upper bound
            ],
            D=[
                np.sqrt(seed_ranges.C[0] * seed_ranges.N_to_D[0] / 6),  # lower bound
                np.sqrt(seed_ranges.C[1] * seed_ranges.N_to_D[1] / 6),  # upper bound
            ],
        )

        # Fit
        self.model_search_config = model_search_config
        self.param_grid = param_grid
        self.loss_fn = loss_fn
        self.weight_fn = weight_fn

        # Scale (optionally preset, optionally overridden)
        self.num_seeding_steps = num_seeding_steps  # Only for when using `step` shorthand
        self.scaling_factor = scaling_factor

        """Initialize/load the scaling database & visualizer"""
        self.project_dir = project_dir
        if os.path.isfile(self.project_dir):
            raise FileExistsError(f'There is a file matching the specified directory name "{self.project_dir}"')
        elif not os.path.exists(self.project_dir):
            self.logger.info(f"Creating a directory: [u]{project_dir}[/]")
            os.mkdir(self.project_dir)
        else:
            self.logger.info(f"Mounting a directory: [u]{self.project_dir}[/]")

        self.database = Database(self.project_dir, log_level=log_level)
        self.visualizer = Visualizer(self.project_dir, log_level=log_level)
        self._create_shortcuts()
        self.logger.info("Chinchilla instance created successfully.")

    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> Chinchilla:
        """
        Constructs a Chinchilla instance from a configuration file, with the option to override specific settings.

        Args:
            config_path (str): The path to the configuration file in JSON or YAML format.
            **kwargs: Optional keyword arguments to override configuration settings.

        Returns:
            Chinchilla: A new instance of Chinchilla configured based on the provided file and overrides.

        Raises:
            ValueError: If the configuration file format is not supported.
        """
        with open(config_path, "r") as file:
            if config_path.lower().endswith(".json"):
                import json

                config = json.load(file)
            elif config_path.lower().endswith((".yaml", ".yml")):
                # import yaml  # Not compatible with the scientific notation like 1e18
                # config = yaml.safe_load(file)
                from ruamel.yaml import YAML

                yaml = YAML(typ="safe")

                config = yaml.load(file)
            else:
                raise ValueError("Unsupported file format. Please use JSON or YAML.")

        config.update(kwargs)
        return cls(**config)

    def _create_shortcuts(self) -> None:
        """Sets up shortcut methods."""
        # Bypass instance methods to class methods; override the class method once constructed
        self.allocate_compute = lambda C: Chinchilla.allocate_compute(C, self.get_params())
        self.predict_loss = lambda N, D: Chinchilla.predict_loss(N, D, self.get_params())

        # Submodules; consult each class for what it does
        self.append = self.database.append
        self.plot = lambda *args, **kwargs: self.visualizer.plot(self, *args, **kwargs)

        # Self-shorthands
        self.allocate = self.allocate_compute
        self.L = self.predict_loss
        self.__param_grid_keys = set(self.param_grid.keys())

    # Simulation utility
    def simulate(self, *args, **kwargs) -> None:
        """
        Simulates the scaling law estimation process using the provided arguments.
        This method is a wrapper around the Simulator class, allowing for quick setup and execution of simulations.

        Args:
            *args: Variable length argument list to be passed to `Simulator.__call__`.
            **kwargs: Arbitrary keyword arguments to be passed to `Simulator.__call__`.
        """
        from .simulator import Simulator

        Simulator(self)(*args, **kwargs)

    def seed(self) -> tuple[tuple[int, float], dict[str, int] | None]:
        """
        Sample a random allocation and model configuration from the user-specified seed regime.

        Returns:
            `(N, D), model_config` - A tuple containing the allocation $(N, D)$ followed by
            a model configuration dictionary corresponding to $N$. If `model_search_config` is not specified,
            the latter will be `None`.


        Raises:
            ValueError: If a valid configuration could not be found after a certain number of trials.
        """
        get_model_config = self.model_search_config is not None

        _max_iters = 2**10
        for i in range(_max_iters):
            C = np.exp(random.uniform(*np.log(self.seed_ranges.C)))
            ratio_N_to_D = np.exp(random.uniform(*np.log(self.seed_ranges.N_to_D)))
            N = np.sqrt(C / (6 * ratio_N_to_D))  # N^2 = C / (6*(D/N))
            D = N * ratio_N_to_D
            if get_model_config:
                model_config, N = search_model_config(N, **self.model_search_config, use_cache=i)
                C = 6 * N * D
            else:
                model_config = None
            if is_between(C, self.seed_ranges.C) and is_between(D / N, self.seed_ranges.N_to_D):
                break
        else:
            raise ValueError(f"We could not find a valid configuration in {_max_iters} trials.")

        self.logger.debug(f"[{ordinal(len(self.database.df)+1)}]\t{C:.2e} FLOPs => {N:.2e} params * {D:.2e} samples")

        return (N, D), model_config

    def fit(self, parallel: bool = True, simulation: bool = False) -> None:
        """
        Uses [L-BFGS optimization (SciPy implementation)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
        to find the best-fitting parameters for the scaling law based on the collected data.

        Args:
            parallel (bool, optional): Whether to run L-BFGS optimization over the initialization grid in parallel processing.
            simulation (bool, optional): Indicates whether the fitting is part of a simulation. Defaults to False.

        Raises:
            ValueError: If there are not enough data points to perform the fitting.
            TypeError: If the numerical precision is insufficient for the L-BFGS algorithm.
            NotImplementedError: When you try to use `weight_fn` for the first time; you are supposed to start _hacking_ here.
        """
        _df = self.database.df.copy()

        if len(_df) < 3:
            raise ValueError(
                f"You do not have enough number of training runs on the database yet. Currently: {len(_df)}"
            )

        if DTYPE().itemsize < 8:  # In bytes
            raise TypeError(
                "The current operation requires a numerical precision of at least 64-bit as used in the L-BFGS algorithm. Lower precisions such as np.float32 or below are not supported for this operation. Please ensure you're using np.float64 or higher precision to avoid this error."
            )

        # Pre-compute the series repeatedly accessed by `self._evaluate_params`
        if self.weight_fn:
            raise NotImplementedError(
                "When specifying `weght_fn, you are expected to edit the source code by deleting this error and specify how to compute yourself."
            )
            weights = self.weight_fn(_df.C.values.astype(DTYPE))
            weights /= weights.mean()
        else:
            weights = None

        self._const = dict(
            log_N=np.log(_df.N.values.astype(DTYPE)),
            log_D=np.log(_df.D.values.astype(DTYPE)),
            y_true=_df.loss.values.astype(DTYPE),
            weights=weights,
        )

        # The absolute value range affects the differential optimization
        self._autoscale_range = np.array(list(map(np.ptp, self.param_grid.values())))
        # In case of any axis with a single initial value:
        self._autoscale_range[self._autoscale_range == 0] = 1.0

        if parallel:
            # GLobal declarations for `multiprocessing`
            global initial_guesses
            global _optimize_params

        initial_guesses = list(itertools.product(*self.param_grid.values()))
        initial_guesses /= self._autoscale_range

        def _optimize_params(i):
            x0 = initial_guesses[i]
            # res = sciop.minimize(self._evaluate_params, x0, method="L-BFGS-B", tol=1e-7)  # Note: `tol` -> `ftol`
            # lbfgs_loss = res.fun
            # if np.isfinite(lbfgs_loss):
            #     return res.x * self._autoscale_range, lbfgs_loss
            # https://github.com/scipy/scipy/blob/v1.12.0/scipy/optimize/_lbfgsb_py.py
            x, lbfgs_loss, _ = sciop.fmin_l_bfgs_b(
                self._evaluate_params,
                x0,
                approx_grad=True,
                maxiter=1_000_000,
                maxfun=1_000_000,
                # Default values generally perform fine
            )
            if np.isfinite(lbfgs_loss):
                return x, lbfgs_loss

        with multiprocessing.Pool(os.cpu_count()) as pool:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                "/",
                TimeRemainingColumn(),
                disable=self.logger.getEffectiveLevel() > 30,
            ) as progress:
                task = progress.add_task(
                    ("" if not simulation else "[SIMULATION] ") + "Fitting scaling law", total=len(initial_guesses)
                )
                results = []
                if parallel:
                    self.logger.debug(f"{os.cpu_count()=}")
                    for res in pool.imap_unordered(_optimize_params, range(len(initial_guesses))):
                        if res:
                            results.append(res)
                        progress.update(task, advance=1.0)
                else:
                    for i in range(len(initial_guesses)):
                        res = _optimize_params(i)
                        if res:
                            results.append(res)
                        progress.update(task, advance=1.0)

        if not results:
            raise ValueError("No valid result from L-BFGS. `loss_fn` you have specified is possibly broken.")
        best_fit = min(results, key=lambda x: x[1])
        self.E, self.A, self.B, self.alpha, self.beta = best_fit[0] * self._autoscale_range

        # Adjust and update if any of the user-specified scales were logarithms
        for param_name in ("e", "a", "b"):
            if param_name in self.__param_grid_keys:
                setattr(self, param_name.upper(), np.exp(getattr(self, param_name.upper()), dtype=DTYPE))

        N, D = _df.N.values, _df.D.values
        # N, D = N.astype(float), D.astype(float)  # In case N and D were type object in pandas/numpy
        y_pred = self.predict_loss(N, D)
        self.visualizer.LBFGS(y_pred, _df.loss.values, simulation=simulation)

        self.logger.info(
            f"Loss predictor:\n\n  L(N, D) = {self.E:#.4g} + {self.A:#.4g} / (N ^ {self.alpha:#.4g}) + {self.B:#.4g} / (D ^ {self.beta:#.4g})\n"
        )

    def scale(
        self,
        scaling_factor: float | None = None,  # Optional override
        C: float | None = None,  # Directly specify the compute budget for the step
        simulation: bool = False,
    ) -> tuple[tuple[int, float], dict[str, int] | None]:
        """
        Determines the compute-optimal allocation of a scaled FLOP budget for the next model.

        > **Example Usages**:
        >
        > 1. Specifying/overriding `scaling_factor` in place
        > ```diff
        > cc = Chinchilla(
        >     ...
        > -    scaling_factor=10.0,
        >     ...
        > )
        > for i in range(100 + 8):
        >     ...
        >     cc.fit()
        > -    (N, D), model_config = cc.scale()
        > +    (N, D), model_config = cc.scale(scaling_factor=10.0)
        > ```
        >
        > 2. Directly specify the FLOP budget
        > ```python
        > (N, D), model_config = cc.scale(C=5.76e23)
        > ```

        Args:
            scaling_factor (float | None, optional): An optional scaling factor to override the instance's scaling factor. Defaults to None.
            C (float | None, optional): Directly specify the compute budget for the scaling step. Defaults to None.
            simulation (bool, optional): Indicates whether the scaling is part of a simulation. Defaults to False.

        Returns:
            `(N, D), model_config` - A tuple containing the allocation $(N, D)$ and
            an optional dictionary with the model configuration corresponding to $N$.

        Raises:
            ValueError: If any of the following conditions are met:
                - The scaling law parameters have not been estimated
                - neither `C` nor `scaling_factor` are specified
                - both `C` and `scaling_factor` are specified
        """
        get_model_config = self.model_search_config is not None

        if not all(hasattr(self, param) for param in ["alpha", "beta"]):
            raise ValueError("You must call `fit` before training a model with scaled compute.")

        if not any([C, scaling_factor, self.scaling_factor]):
            raise ValueError("You must specify either `C` or `scaling_factor`")

        if all([C, scaling_factor]):
            raise ValueError("You cannot specify both `C` and `scaling_factor`")

        # Prioritize parameters in order of C, scaling_factor, self.scaling_factor
        if C is None:
            # Use the preset `scaling_factor` if not overridden
            scaling_factor = scaling_factor or self.scaling_factor
            C = max(self.seed_ranges.C[1], int(self.database.df.C.max())) * scaling_factor

        N, D = self.allocate_compute(C)
        if get_model_config:
            model_config, N = search_model_config(N, **self.model_search_config)
            D = self.adjust_D_to_N(N)
            C = 6 * N * D
            if C <= int(self.database.df.C.max()):
                self.logger.warning("The scaling process has saturated. Consider increasing some numbers")
        else:
            model_config = None

        self.logger.info(f"[{ordinal(len(self.database.df)+1)}]\t{C:.2e} FLOPs => {N:.2e} params * {D:.2e} samples")
        self.plot(next_point=dict(C=C, N=N, D=D), simulation=simulation)

        return (N, D), model_config

    def step(
        self, num_seeding_steps: int | None = None, parallel: bool = True, simulation: bool = False, **scale_kwargs
    ) -> tuple[tuple[int, float], dict[str, int] | None]:
        """
        Shorthand method automatically routing to `seed` or `fit` & `scale` methods,
        depending on the existing number of training runs in the seed regime.

        > If you prefer to be explicit about the seeding and scaling steps, you can use the following approach:
        >
        > ```diff
        > - (N, D), model_config = cc.step(num_seeding_steps=401)
        > + if len(cc.database.df) < 401:
        > +     (N, D), model_config = cc.seed()
        > + else:
        > +     cc.fit_scaling_law()
        > +     (N, D), model_config = cc.scale()
        > ```

        Args:
            num_seeding_steps (int, optional): The threshold number of seed training runs before starting to scale the compute budget.
            parallel (bool, optional): Whether to run L-BFGS optimization over the initialization grid in parallel processing. To be passed to `fit`.
            simulation (bool, optional): Indicates whether the scaling is part of a simulation. Defaults to False.
            **scale_kwargs: Keyword arguments to be passed to `scale` (`scaling_factor` and `C`).

        Returns:
            `(N, D), model_config` - A tuple containing the allocation $(N, D)$ and
            an optional dictionary with the model configuration corresponding to $N.

        Raises:
            ValueError: If any of the following conditions are met:
                - `num_seeding_steps` is not specified
                - neither `C` nor `scaling_factor` are specified
                - both `C` and `scaling_factor` are specified
        """
        # Use the preset value if not overridden
        num_seeding_steps = num_seeding_steps or self.num_seeding_steps

        if num_seeding_steps is None:
            raise ValueError(
                "To use `step` shorthand method, you must specify `num_seeding_steps` either in `__init__` or this method"
            )

        num_valid_seeds = self.database.df.C.apply(lambda x: is_between(x, self.seed_ranges.C)).sum()
        if num_valid_seeds < num_seeding_steps:
            if not simulation:
                self.logger.info(
                    textwrap.dedent(
                        f"""\
                       Existing number of [i]seed[/] training runs: {num_valid_seeds}
                    -> Sample a model configuration (N, D) from the seed regime"""
                    )
                )
            return self.seed()
        else:
            # pre-validate scale_kwargs
            if not any([hasattr(scale_kwargs, "C"), hasattr(scale_kwargs, "scaling_factor"), self.scaling_factor]):
                raise ValueError("You must specify either `C` or `scaling_factor` as a positive number")

            if hasattr(scale_kwargs, "C") and hasattr(scale_kwargs, "scaling_factor"):
                raise ValueError("You cannot specify both `C` and `scaling_factor`")

            if not simulation:
                self.logger.info(
                    textwrap.dedent(
                        f"""\
                           Existing number of [i]seed[/] training runs: {num_valid_seeds}
                        -> Estimate the parameters ([i]A[/], [i]B[/], [i]α[/], [i]β[/]) from {len(self.database.df)} training runs"""
                    )
                )
            self.fit(simulation=simulation, parallel=parallel)
            return self.scale(simulation=simulation, **scale_kwargs)

    def adjust_D_to_N(self, N: float) -> float:
        r"""
        Adjusts $D$ (the number of data samples) to $N$ (the number of model parameters) based on the scaling law.
        Computes:

            $$D = G^{-(1 + b/a)} N^{b/a}$$

        > **Example Usage**:
        > ```python
        > (N, D), model_config = cc.scale()
        > model = Model(**model_config)
        > N = sum(p.numel() for p in model.parameters())
        > D = cc.adjust_D_to_N(N)
        > ```
        > Once you get an estimate of the scaling law for your task,
        > you may want to update $D$ to match the actual value of $N$ if your `estimate_model_size` is not strictly accurate.


        Args:
            N (float): The number of model parameters.

        Returns:
            float: The adjusted number of data samples.

        Raises:
            ValueError: If N is not a positive number.
        """
        if not isinstance(N, (int, float)) or N <= 0:
            raise ValueError(f"N must be a positive number, but got {N}")

        alpha_beta = self.alpha + self.beta
        G = np.power((self.alpha * self.A) / (self.beta * self.B), (1 / alpha_beta))
        _a, _b = self.beta / alpha_beta, self.alpha / alpha_beta
        D = np.power(G, -1 - _b / _a) * np.power(N, _b / _a)

        self.logger.debug(f"Adjusted D to N: {D=:.1f} | {N=}")
        return D

    @classmethod
    def allocate_compute(cls, C: float | list | np.ndarray, params: dict) -> tuple[float, float] | np.ndarray:
        r"""
        Allocates a given computational budget (C) to the optimal number of model parameters (N) and data samples (D),
        which wouls satisfy the following formula based on the scaling law parameters provided in the `params` dictionary.

        $$\underset{N,\ D}{argmin}\ L(N,\ D\ |\ E,\ A,\ B,\ \alpha,\ \beta)$$

        Once instantiated, this class method gets overridden by `__allocate_compute` so that `params` are
        automatically specified from the instance attributes.

        > **Example Usages**:
        > 1. As a class method
        > ```python
        > params = {
        >     "E":      1.620406544125793,
        >     "A":      1116.7583712076722171,
        >     "B":      92697.423904473161286,
        >     "alpha":  0.6491512524478403,
        >     "beta":   0.7105431526502198,
        > }
        > N, D = Chinchilla.allocate_compute(1e18, params)
        > ```
        >
        > 2. As an instance method
        > ```python
        > cc = Chinchilla(...)
        > cc.fit()  # internally or explicitly
        > N, D = cc.allocate_compute(np.logspace(18, 21))
        > ```

        Args:
            C (float | list | np.ndarray): The computational budget in FLOPs. Can be a single value or an array.
            params (dict): A dictionary containing the scaling law parameters (alpha, beta, A, B).

        Returns:
            tuple[float, float] | np.ndarray: A tuple containing the optimal number of model parameters (N) and
            data samples (D). If C is an array, the output will be a 2D array with shape (len(C), 2).

        Raises:
            ValueError: If `params` is missing any of the required parameters (alpha, beta, A, B).
        """
        required_params = ["alpha", "beta", "A", "B"]
        if not all(param in params for param in required_params):
            raise ValueError(f"Missing required parameters. Expected: {required_params}")
        alpha, beta, A, B = [params[param] for param in required_params]

        alpha_beta = alpha + beta
        G = np.power((alpha * A) / (beta * B), (1 / alpha_beta))
        _a, _b = beta / alpha_beta, alpha / alpha_beta

        N_opt = G * np.power(C / 6, _a)
        D_opt = (1 / G) * np.power(C / 6, _b)

        return N_opt, D_opt

    @classmethod
    def predict_loss(cls, N: np.ndarray | float, D: np.ndarray | float, params: dict) -> np.ndarray | float:
        r"""
        Predicts the loss for given allocations of model parameters (N) and data samples (D) using the scaling law
        parameters provided in the `params` dictionary.

        The loss is calculated based on the following formula:
        $$L(N,\ D\ |\ E,\ A,\ B,\ \alpha,\ \beta) = E + A \cdot N^{-\alpha} + B \cdot D^{-\beta}$$

        Once instantiated, this class method gets overridden by `__predict_loss` so that `params` are
        automatically specified from the instance attributes.

        > **Example Usages**:
        > 1. As a class method
        > ```python
        > params = {
        >     "E":      1.620406544125793,
        >     "A":      1116.7583712076722171,
        >     "B":      92697.423904473161286,
        >     "alpha":  0.6491512524478403,
        >     "beta":   0.7105431526502198,
        > }
        > N, D = 1e9, 1e6
        > loss = Chinchilla.predict_loss(N, D, params)
        > ```
        >
        > 2. As an instance method
        > ```python
        > cc = Chinchilla(...)
        > cc.fit()  # internally or explicitly
        > N, D = np.array([1e9, 1e10]), np.array([1e6, 1e7])
        > loss = cc.predict_loss(N, D)
        > ```

        Args:
            N (np.ndarray | float): The number of model parameters or an array of such numbers.
            D (np.ndarray | float): The number of data samples or an array of such numbers.
            params (dict): A dictionary containing the scaling law parameters (E, A, B, alpha, beta).

        Returns:
            np.ndarray | float: The predicted loss or an array of predicted losses.

        Raises:
            ValueError: If `params` is missing any of the required parameters (E, A, B, alpha, beta).
        """
        required_params = ["E", "A", "B", "alpha", "beta"]
        if not all(param in params for param in required_params):
            raise ValueError(f"Missing required parameters. Expected: {required_params}")

        E, A, B, alpha, beta = [params[param] for param in required_params]

        log_term_2nd = np.log(A) - alpha * np.log(N)
        log_term_3rd = np.log(B) - beta * np.log(D)

        return E + np.exp(log_term_2nd) + np.exp(log_term_3rd)

    def _evaluate_params(self, x) -> np.ndarray:
        """
        Internal method to compute the loss for the L-BFGS algorithm.

        This method evaluates the loss function for a given set of parameters during the optimization process.

        Args:
            X (np.ndarray): The array of parameters to evaluate.

        Returns:
            float: The computed loss value.

        Raises:
            ValueError: If NaN values are detected in the input parameters.
        """
        if np.isnan(x).any():
            raise ValueError(
                f"NaN value(s) detected in input to `cc._evaluate_params`: {x}\n"
                f"This was possibly because `loss_fn` you specified is compatible with type `{DTYPE}`"
            )

        # Scipy/Fortran implementation of LBFGS casts `x0` to float64 internally, so recover here.
        # Invert autoscaling & decompose
        E, a, b, alpha, beta = x.astype(DTYPE) * self._autoscale_range

        # Ensure the log scale for `a` and `b` but `E`.
        # Inspect the user-specified initial keys requiring `exp`/`log` transformation
        if "e" in self.__param_grid_keys:
            E = np.exp(E)
        if "A" in self.__param_grid_keys:
            a = np.log(np.clip(a, FLOAT_TINY, None))
        if "B" in self.__param_grid_keys:
            b = np.log(np.clip(b, FLOAT_TINY, None))

        log_term_2nd = a - alpha * self._const["log_N"]
        log_term_3rd = b - beta * self._const["log_D"]
        log_term_2nd = np.clip(log_term_2nd, None, FLOAT_LOGMAX)
        log_term_3rd = np.clip(log_term_3rd, None, FLOAT_LOGMAX)
        y_pred = E + np.exp(log_term_2nd) + np.exp(log_term_3rd)

        losses = self.loss_fn(self._const["y_true"], y_pred)

        if self.weight_fn:
            losses = losses * self._const["weights"]

        return np.mean(losses)

    def get_params(self) -> dict:
        """
        Returns a dictionary of estimated parameters describing the scaling law / parametric loss estimator.

        Returns:
            float: The computed loss value.

        Raises:
            ValueError: If the scaling law parameters have not been set as attributes.
        """
        if not all(hasattr(self, param) for param in ["E", "A", "B", "alpha", "beta"]):
            raise ValueError("You must call `fit` before training a model with scaled compute.")
        return {"E": self.E, "A": self.A, "B": self.B, "alpha": self.alpha, "beta": self.beta}

    def report(self, plot: bool = True) -> None:
        """
        Generates a report summarizing the scaling law estimation results.

        The report includes:
        - Estimated scaling law parameters (E, A, B, alpha, beta)
        - Goodness of fit with regards to specified loss function
        - Plots of the actual vs. predicted losses
        - Suggested compute allocation for the next model based on the scaling law

        Raises:
            ValueError: If the scaling law parameters have not been estimated yet.
        """
        if not all(hasattr(self, param) for param in ["alpha", "beta"]):
            raise ValueError("You must call `fit` before generating a report.")

        self.logger.info("Estimated scaling law parameters:")
        for param, value in self.get_params().items():
            self.logger.info(f"    - {param}: {value}")
        if len(self.database.df):
            self.logger.info("Goodness of fit:")
            y_true = self.database.df.loss.values
            y_pred = self.predict_loss(self.database.df.N.values, self.database.df.D.values)
            weights = 1.0
            if self.weight_fn:
                weights = self.weight_fn(self.database.df.C.values.astype(DTYPE))
                weights /= weights.mean()
            loss = (self.loss_fn(y_true, y_pred) * weights).mean()
            self.logger.info(f"    - `{self.loss_fn.__name__}`: {loss:.8f}")
            if plot:
                self.logger.info("Landscape visualization:")
                self.plot()
