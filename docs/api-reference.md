# API Reference

# chinchilla.core.Chinchilla

```python
class Chinchilla()
```

Estimates the scaling law for a deep learning task. Provides functionalities to:

1. Sample models from a specified "seed" regime.
2. Fit the loss predictor $L(N, D)$.
3. Suggest an allocation of scaled compute.

This module includes the `Chinchilla` class, which provides methods for sampling model configurations, fitting the parametric loss predictor, suggesting allocations for scaled compute budgets, etc. It operates in a numerical precision of **128-bit** by default and integrates with [`chinchilla.Database`](#chinchilladatabaseDatabase) and [`chinchilla.Visualizer`](#chinchillavisualizerVisualizer) for storing and plotting data.

#### \_\_init\_\_

```python
def __init__(project_dir: str,
             param_grid: dict[str, np.ndarray | list | tuple],
             seed_ranges: dict[str, np.ndarray | list | tuple],
             model_search_config: dict[str, Callable | dict] | None = None,
             loss_fn: Callable = asymmetric_mae,
             weight_fn: Callable | None = None,
             num_seeding_steps: int | None = None,
             scaling_factor: float | None = None,
             log_level: int | str = 20) -> None
```

Initializes a Chinchilla instance with the given parameters and sets up the scaling process.

**Arguments**:

- `project_dir` _str_ - The directory path for the project where the database and visualizations will be stored.
- `param_grid` _dict[str, np.ndarray | list | tuple]_ - A dictionary specifying the grid of parameters to search over.
- `seed_ranges` _dict[str, np.ndarray | list | tuple]_ - A dictionary specifying the ranges for seeding the model configurations.
- `model_search_config` _dict[str, Callable | dict] | None, optional_ - Configuration for model search. Defaults to None.
- `loss_fn` _Callable, optional_ - The loss function to be used for fitting. Defaults to asymmetric_mae.
- `weight_fn` _Callable | None, optional_ - A function to weight loss prediction errors. Defaults to None.
- `num_seeding_steps` _int | None, optional_ - The number of seeding steps to perform. Defaults to None.
- `scaling_factor` _float | None, optional_ - The scaling factor to be used when scaling up compute. Defaults to None.
- `log_level` _int | str, optional_ - Specifies the threshold for logging messages. A value of 30 suppresses standard messages while any larger values hide all messages entirely. Defaults to 20 (`logging.INFO`).

**Raises**:

- `ValueError` - If `project_dir` does not exist or is not a directory.
- `TypeError` - If `loss_fn` or `weight_fn` is not callable.
- `FileExistsError` - If a file with the same name as `project_dir` already exists.

#### from_config

```python
@classmethod
def from_config(cls, config_path: str, **kwargs) -> Chinchilla
```

Constructs a Chinchilla instance from a configuration file, with the option to override specific settings.

**Arguments**:

- `config_path` _str_ - The path to the configuration file in JSON or YAML format.
- `**kwargs` - Optional keyword arguments to override configuration settings.

**Returns**:

- `Chinchilla` - A new instance of Chinchilla configured based on the provided file and overrides.

**Raises**:

- `ValueError` - If the configuration file format is not supported.

#### simulate

```python
def simulate(*args, **kwargs) -> None
```

Simulates the scaling law estimation process using the provided arguments. This method is a wrapper around the Simulator class, allowing for quick setup and execution of simulations.

**Arguments**:

- `*args` - Variable length argument list to be passed to `Simulator.__call__`.
- `**kwargs` - Arbitrary keyword arguments to be passed to `Simulator.__call__`.

#### seed

```python
def seed() -> tuple[tuple[int, float], dict[str, int] | None]
```

Sample a random allocation and model configuration from the user-specified seed regime.

**Returns**:

`(N, D), model_config` - A tuple containing the allocation $(N, D)$ followed by a model configuration dictionary corresponding to $N$. If `model_search_config` is not specified, the latter will be `None`.

**Raises**:

- `ValueError` - If a valid configuration could not be found after a certain number of trials.

#### fit

```python
def fit(parallel: bool = True, simulation: bool = False) -> None
```

Uses [L-BFGS optimization (SciPy implementation)](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) to find the best-fitting parameters for the scaling law based on the collected data.

**Arguments**:

- `parallel` _bool, optional_ - Whether to run L-BFGS optimization over the initialization grid in parallel processing.
- `simulation` _bool, optional_ - Indicates whether the fitting is part of a simulation. Defaults to False.

**Raises**:

- `ValueError` - If there are not enough data points to perform the fitting.
- `TypeError` - If the numerical precision is insufficient for the L-BFGS algorithm.
- `NotImplementedError` - When you try to use `weight_fn` for the first time; you are supposed to start _hacking_ here.

#### scale

```python
def scale(
    scaling_factor: float | None = None,
    C: float | None = None,
    simulation: bool = False
) -> tuple[tuple[int, float], dict[str, int] | None]
```

Determines the compute-optimal allocation of a scaled FLOP budget for the next model.

> **Example Usages**:
>
> 1. Specifying/overriding `scaling_factor` in place
>
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
>
> ```python
> (N, D), model_config = cc.scale(C=5.76e23)
> ```

**Arguments**:

- `scaling_factor` _float | None, optional_ - An optional scaling factor to override the instance's scaling factor. Defaults to None.
- `C` _float | None, optional_ - Directly specify the compute budget for the scaling step. Defaults to None.
- `simulation` _bool, optional_ - Indicates whether the scaling is part of a simulation. Defaults to False.

**Returns**:

`(N, D), model_config` - A tuple containing the allocation $(N, D)$ and an optional dictionary with the model configuration corresponding to $N$.

**Raises**:

- `ValueError` - If any of the following conditions are met:
  - The scaling law parameters have not been estimated
  - neither `C` nor `scaling_factor` are specified
  - both `C` and `scaling_factor` are specified

#### step

```python
def step(num_seeding_steps: int | None = None,
         parallel: bool = True,
         simulation: bool = False,
         **scale_kwargs) -> tuple[tuple[int, float], dict[str, int] | None]
```

Shorthand method automatically routing to `seed` or `fit` & `scale` methods, depending on the existing number of training runs in the seed regime.

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

**Arguments**:

- `num_seeding_steps` _int, optional_ - The threshold number of seed training runs before starting to scale the compute budget.
- `parallel` _bool, optional_ - Whether to run L-BFGS optimization over the initialization grid in parallel processing. To be passed to `fit`.
- `simulation` _bool, optional_ - Indicates whether the scaling is part of a simulation. Defaults to False.
- `**scale_kwargs` - Keyword arguments to be passed to `scale` (`scaling_factor` and `C`).

**Returns**:

`(N, D), model_config` - A tuple containing the allocation $(N, D)$ and an optional dictionary with the model configuration corresponding to $N.

**Raises**:

- `ValueError` - If any of the following conditions are met:
  - `num_seeding_steps` is not specified
  - neither `C` nor `scaling_factor` are specified
  - both `C` and `scaling_factor` are specified

#### adjust_D_to_N

```python
def adjust_D_to_N(N: float) -> float
```

Adjusts $D$ (the number of data samples) to $N$ (the number of model parameters) based on the scaling law. Computes:

$$D = G^{-(1 + b/a)} N^{b/a}$$

> **Example Usage**:
>
> ```python
> (N, D), model_config = cc.scale()
> model = Model(**model_config)
> N = sum(p.numel() for p in model.parameters())
> D = cc.adjust_D_to_N(N)
> ```
>
> Once you get an estimate of the scaling law for your task, you may want to update $D$ to match the actual value of $N$ if your `estimate_model_size` is not strictly accurate.

**Arguments**:

- `N` _float_ - The number of model parameters.

**Returns**:

- `float` - The adjusted number of data samples.

**Raises**:

- `ValueError` - If N is not a positive number.

#### allocate_compute

```python
def allocate_compute(
        C: float | list | np.ndarray) -> tuple[float, float] | np.ndarray
```

Allocates a given computational budget (C) to the optimal number of model parameters (N) and data samples (D):

$$\underset{N,\ D}{argmin}\ L(N,\ D\ |\ E,\ A,\ B,\ \alpha,\ \beta)$$

**Arguments**:

- `C` _float_ - The computational budget in FLOPs.

**Returns**:

tuple[float, float]: A tuple containing the optimal number of model parameters (N) and data samples (D).

**Raises**:

- `ValueError` - If C is not a positive number.

#### predict_loss

```python
def predict_loss(N: np.ndarray | float,
                 D: np.ndarray | float) -> np.ndarray | float
```

Predicts the loss for given allocations of model parameters (N) and data samples (D).

**Arguments**:

- `N` _np.ndarray | float_ - The number of model parameters or an array of such numbers.
- `D` _np.ndarray | float_ - The number of data samples or an array of such numbers.

**Returns**:

_np.ndarray | float_: The predicted loss or an array of predicted losses.

#### get_params

```python
def get_params(self) -> dict
```

Returns a dictionary of estimated parameters describing the scaling law / parametric loss estimator.

**Returns**:

dict: A dictionary containing the estimated scaling law parameters.

**Raises**:

- `ValueError` - If the scaling law parameters have not been estimated.

#### report

```python
def report(self, plot: bool = True) -> None
```

Generates a report summarizing the scaling law estimation results.

The report includes:

- Estimated scaling law parameters (E, A, B, alpha, beta)
- Goodness of fit with regards to specified loss function
- Plots of the actual vs. predicted losses
- Suggested compute allocation for the next model based on the scaling law

**Arguments**:

- `plot` _bool, optional_ - Whether to include plots in the report. Defaults to True.

**Raises**:

- `ValueError` - If the scaling law parameters have not been estimated yet.

# chinchilla.database.Database

```python
class Database()
```

Stores and manipulates scaling data in a Pandas DataFrame the default persistence to a CSV file. The Database class is used internally by a `Chinchilla` instance.

If `project_dir` is provided, the DataFrame is initialized from the CSV file at that location. If the file does not exist or is empty, a new DataFrame is created. If `project_dir` is None, the DataFrame is kept in memory.

**Default columns**:

- `C` (float): Compute in FLOPs.
- `N` (int): Number of parameters.
- `D` (int): Number of data samples seen.
- `loss` (float): Loss value (optional, use case dependent).

**Arguments**:

- `project_dir` _str | None_ - Directory for the CSV file storage.
- `columns` _list[str]_ - Column names for the DataFrame.
- `logger` _Logger_ - Logger instance for database messages.

#### \_\_init\_\_

```python
def __init__(project_dir: str | None = None,
             columns: list[str] = ["C", "N", "D", "loss"],
             log_level: int = 30) -> None
```

Initializes the Database instance.

**Arguments**:

- `project_dir` _Optional[str]_ - The directory path to save the DataFrame as a CSV file. If None, the DataFrame will not be saved to disk.
- `columns` _List[str]_ - A list of column names for the DataFrame.
- `log_level` _int_ - The logging level for the logger instance.

#### append

```python
def append(**result: dict[str, float]) -> None
```

Appends a new row of results to the DataFrame and updates the CSV file if `project_dir` is set.

If 'C' is not provided in `result`, it is automatically calculated as $6ND$. All numerical values are rounded to the nearest integer to prevent scientific notation in large values. Additional columns provided by the user are appended to the DataFrame.

**Arguments**:

- `result` _dict[str, float]_ - A dictionary containing the data to append. Must include 'N', 'D', and 'loss' keys. If 'C' is not provided in `result`, it is automatically calculated as $6ND$. All numerical values are rounded to the nearest integer to prevent losing precisions to scientific notation for large values. Additional columns provided by the user will be appended to the DataFrame without any conflicts.

# chinchilla.visualizer.Visualizer

```python
class Visualizer()
```

`Visualizer` includes methods for plotting the estimated loss gradient, the efficient frontier, and L-BFGS optimization results. It helps in understanding the distribution and relationships between compute resources, model parameters, and data samples, and highlights efficient allocation frontiers and seed regimes.

**Attributes**:

- `project_dir` _str_ - Directory to save plots.
- `logger` _Logger_ - Logger instance for visualization messages.
- `cc` _Chinchilla_ - A Chinchilla instance passed through the `plot` method to be shared across its submethods.

#### \_\_init\_\_

```python
def __init__(project_dir: str, log_level: int = 30) -> None
```

Constructs a Visualizer instance with a project directory and logging level.

**Arguments**:

- `project_dir` _str_ - The directory where the plots will be saved.
- `log_level` _int_ - Logging level to control the verbosity of log messages.

#### plot

```python
def plot(cc,
         next_point: dict[str, float] | None = None,
         fitted: bool = True,
         img_name: str = "parametric_fit",
         cmap_name: str = "plasma",
         simulation: bool = False) -> None
```

Plots the loss gradient and efficient frontier for resource allocation.

This method visualizes the distribution and relationships between compute resources (FLOPs), model parameters, and data samples. It shows how these factors interact with the loss function and highlights efficient allocation frontiers and seed regimes.

**Example output**: ![](../examples/efficientcube-1e15_1e16/parametric_fit.png)

**Arguments**:

- `cc` - A Chinchilla instance with a Database of training runs and scaling law parameters if estimated.
- `next_point` _dict[str, float] | None_ - The next point to be plotted, if any.
- `fitted` _bool_ - Whether to plot the scaling law gradient or only raw data points. If the loss predictor is not fitted, falls back to False.
- `img_name` _str_ - The name of the image file to save the plot as.
- `cmap_name` _str_ - The name of the colormap to be used for plotting.
- `simulation` _bool_ - Whether the plot is for a simulation.

#### LBFGS

```python
def LBFGS(y_pred: np.ndarray,
          y_true: np.ndarray,
          C: np.ndarray | None = None,
          simulation: bool = False,
          img_name: str = "LBFGS") -> None
```

Plots the results of L-BFGS optimization, including the loss history and prediction accuracy. This method visualizes the predicted values versus the true labels and the error distribution.

**Example output**: ![](../examples/efficientcube-1e15_1e16/LBFGS.png)

**Arguments**:

- `y_pred` _np.ndarray_ - Predicted values by the model.
- `y_true` _np.ndarray_ - True labels or values for comparison with predictions.
- `C` _np.ndarray | None, optional_ - Compute resources (FLOPs) associated with each prediction, if available.
- `simulation` _bool, optional_ - Whether the plot is for a simulation.
- `img_name` _str, optional_ - The name of the image file to save the plot as.

# chinchilla.simulator.Simulator

```python
class Simulator(Chinchilla)
```

Simulates the scaling law estimation with `Chinchilla`, allowing you to understand its behaviors.

Inheriting and extending the `Chinchilla` class with the capacity to simulate seeding and scaling in a hypothetical task, `Simulator` models how factors like `Chinchilla` configuration, number of seeds, scaling factor, the noisiness of losses, etc. would confound to affect the stability and the performance of scaling law estimation.

**Attributes**:

- `cc` _Chinchilla_ - The Chinchilla instance with preset configuration attributes like `param_grid`, `seed_ranges`, `loss_fn`, etc.
- `logger` _Logger_ - Logger instance for simulation activities.
- `database` _Database_ - Database instance to record simulation results.

**Methods**:

- `__init__` - Initializes the Simulator with a Chinchilla instance.
- `__getattr__` - Delegates attribute access to the Chinchilla instance.
- `__call__` - Executes the simulation with given parameters.
- `_pseudo_training_run` - Performs a pseudo-training run and records results.

#### \_\_init\_\_

```python
def __init__(cc: Chinchilla)
```

Initialize the Simulator with a Chinchilla instance.

**Arguments**:

- `cc` _Chinchilla_ - An instance of Chinchilla to be used for simulation.

#### \_\_getattr\_\_

```python
def __getattr__(name: str) -> Any
```

Delegate attribute access to the Chinchilla instance.

**Arguments**:

- `name` - The name of the attribute to access.

**Returns**:

- `Any` - The value of the attribute with the specified name from the Chinchilla instance.

#### \_\_call\_\_

```python
def __call__(
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
    noise_generator: Iterator | tuple[Callable, tuple[float, ...]]
    | None = (random.expovariate(10) for _ in iter(int, 1))
) -> None
```

Simulate the compute-scaling on a hypothetical deep learning task with some noise expectable from reality.

**Arguments**:

- `num_seeding_steps` _int_ - The number of seeding steps to simulate.
- `num_scaling_steps` _int_ - The number of scaling steps to simulate.
- `scaling_factor` _float | None, optional_ - The scaling factor to be used in the simulation.
- `target_params` _dict[str, float]_ - A dictionary of target parameters for the simulation.
- `noise_generator` _Iterator | tuple[Callable, tuple[float, ...]] | None, optional_ - A callable or iterator that generates noise to be added to the loss. Defaults to `(random.expovariate(10) for _ in iter(int, 1))`, which generates an exponential distribution averaging at $0.100$.

**Raises**:

- `TypeError` - If the provided `noise_generator` is not an iterator or a tuple with a callable and its arguments.

# chinchilla.\_utils

Utility functions.

#### search_model_config

```python
def search_model_config(
        N_opt: float,
        hyperparam_grid: dict[str, list[float]],
        size_estimator: Callable[..., float],
        use_cache: bool = False) -> tuple[dict[str, float], float]
```

Finds the model configuration that is closest to a given target number of parameters, based on the provided hyperparameter grid and size estimator.

> **Example Usage**:
>
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

**Arguments**:

- `N_opt` _float_ - The target number of parameters for the model configuration.
- `hyperparam_grid` _dict[str, list[float]]_ - A dictionary where keys are hyperparameter names and values are lists of possible values.
- `size_estimator` _Callable[..., float]_ - A callable that takes a model configuration as keyword arguments and returns the estimated size.
- `use_cache` _bool, optional_ - A boolean flag indicating whether to use cached estimates. Default is False.

**Returns**:

A tuple containing the closest model configuration and its estimated size.

**Notes**:

Although very efficient, you should set `use_cache` to True only when `hyperparam_grid` is guaranteed to be consistent; thus, it is disabled by default except for Simulator (x16 faster).

#### is_between

```python
def is_between(value: float | np.ndarray,
               bounds: tuple[float, float] | np.ndarray) -> bool | np.ndarray
```

Checks if a value is within the given inclusive bounds.

**Arguments**:

- `value` _float | np.ndarray_ - The value to check.
- `bounds` _tuple[float, float] | np.ndarray_ - A tuple containing the lower and upper bounds.

**Returns**:

bool | np.ndarray: NumPy array: A boolean or an NumPy array of booleans indicating whether the value is between the bounds.

# chinchilla.\_metrics

A few loss & weight functions you can use on demand.

#### asymmetric_mae

```python
def asymmetric_mae(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   w: float = 1e1) -> np.ndarray
```

Asymmetric Mean Absolute Error loss function.

#### huber

```python
def huber(y_true: np.ndarray,
          y_pred: np.ndarray,
          delta: float = 1.0) -> np.ndarray
```

Huber loss function.

#### log_huber

```python
def log_huber(y_true: np.ndarray,
              y_pred: np.ndarray,
              delta: float = 1.0) -> np.ndarray
```

The original loss function used in the Chinchilla paper

#### mae

```python
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Mean Absolute Error loss function.

#### mse

```python
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Mean Squared Error loss function.

# chinchilla.\_logger

Contains a utility function `get_logger`. This module also filters out noisy debug messages from `matplotlib` and suppresses redundant warnings from `numpy` and `matplotlib`.

#### get_logger

```python
def get_logger(level: int | str, name: str) -> logging.Logger
```

Sets up a logger with the specified log level. This logger uses RichHandler for `rich` formatted logging output to the console.

**Arguments**:

- `level` _int | str_ - Logging level, e.g., 20 or logging.INFO, 30 or logging.WARNING.
- `name` _str, optional_ - The name of the logger.

**Returns**:

- `logging.Logger` - Configured logger instance.

# chinchilla.\_validator

Pydantic classes for validating parameters and configurations for `Chinchilla` and `Simulator`.

## ParamGrid

```python
class ParamGrid(BaseModel)
```

Validates a grid of initialization for scaling law (/loss predictor) parameters.

**Attributes**:

E or e: Tuple of floats representing initial values for the E parameter or its log form. A or a: Tuple of floats representing initial values for the A parameter or its log form. B or b: Tuple of floats representing initial values for the B parameter or its log form.

- `alpha` - Tuple of floats representing initial values for the alpha parameter.
- `beta` - Tuple of floats representing initial values for the beta parameter.

#### check_keys

```python
@model_validator(mode="before")
def check_keys(cls, values)
```

Validates that the parameter grid contains the correct keys.

## SeedRanges

```python
class SeedRanges(BaseModel)
```

Validates the regime of seed models.

**Attributes**:

- `C` - Tuple of two floats representing the range for the `C` parameter.
- `N_to_D` - Tuple of two floats representing the range for the `N_to_D` parameter.

#### check_C_values

```python
@field_validator("C")
def check_C_values(cls, values)
```

Validates the values for the 'C' range.

## ModelSearchConfig

```python
class ModelSearchConfig(BaseModel)
```

Validates keyword arguments into `_utils.search_model_config`.

**Attributes**:

- `size_estimator` - A callable that estimates the size of the model.
- `hyperparam_grid` - A dictionary representing the hyperparameter grid.

#### size_estimator_must_be_callable

```python
@field_validator("size_estimator")
def size_estimator_must_be_callable(cls, v)
```

Validates that the size estimator is a callable.

#### hyperparam_grid_must_be_dict

```python
@field_validator("hyperparam_grid")
def hyperparam_grid_must_be_dict(cls, v)
```

Validates that the hyperparameter grid is a dictionary.

## SimulationArgs

```python
class SimulationArgs(BaseModel)
```

Validates arguments for running simulations.

**Attributes**:

- `num_seeding_steps` - The number of seeding steps.
- `num_scaling_steps` - The number of scaling steps.
- `target_params` - A dictionary representing hypothetical scaling law parameters.
- `noise_generator` - An optional iterable of floats representing the additional loss randomly caused by imperfect training.
- `scaling_factor` - An optional float representing the scaling factor in FLOPs.

#### check_seeding_steps

```python
@model_validator(mode="before")
def check_seeding_steps(cls, values)
```

Validates the number of seeding and scaling steps along with the scaling factor.

#### check_target_params

```python
@field_validator("target_params")
def check_target_params(cls, v)
```

Validates that the target parameters are a dictionary with float or int values.

#### check_noise_generator

```python
@field_validator("noise_generator")
def check_noise_generator(cls, v)
```

Validates the noise generator, ensuring it is an iterator or None.
