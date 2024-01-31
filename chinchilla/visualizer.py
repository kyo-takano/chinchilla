from __future__ import annotations  # PEP 604 backport

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._logger import get_logger

sns.set_theme(style="ticks")

PADDING = 0.10


class Visualizer:
    """
    `Visualizer` includes methods for plotting the estimated loss gradient, the efficient frontier,
    and L-BFGS optimization results. It helps in understanding the distribution and relationships between
    compute resources, model parameters, and data samples, and highlights efficient allocation frontiers
    and seed regimes.

    Attributes:
        project_dir (str): Directory to save plots.
        logger (Logger): Logger instance for visualization messages.
        cc (Chinchilla): A Chinchilla instance passed through the `plot` method to be shared across its submethods.
    """

    def __init__(self, project_dir: str, log_level: int = 30) -> None:
        """
        Constructs a Visualizer instance with a project directory and logging level.

        Args:
            project_dir (str): The directory where the plots will be saved.
            log_level (int): Logging level to control the verbosity of log messages.
        """
        self.project_dir = project_dir
        self.logger = get_logger(log_level, name="chinchilla.visualizer")

    def plot(
        self,
        cc,
        next_point: dict[str, float] | None = None,
        fitted: bool = True,
        img_name: str = "parametric_fit",
        cmap_name: str = "plasma",
        simulation: bool = False,
    ) -> None:
        """
        Plots the loss gradient and efficient frontier for resource allocation.

        This method visualizes the distribution and relationships between compute resources
        (FLOPs), model parameters, and data samples. It shows how these factors interact
        with the loss function and highlights efficient allocation frontiers and seed regimes.

        **Example output**:
        ![](../examples/efficientcube-1e15_1e16/parametric_fit.png)

        Args:
            cc: A Chinchilla instance with a Database of training runs and scaling law parameters if estimated.
            next_point (dict[str, float] | None): The next point to be plotted, if any.
            fitted (bool): Whether to plot the scaling law gradient or only raw data points.
                If the loss predictor is not fitted, falls back to False.
            img_name (str): The name of the image file to save the plot as.
            cmap_name (str): The name of the colormap to be used for plotting.
            simulation (bool): Whether the plot is for a simulation.
        """
        # Validate inputs
        if not isinstance(cc.database.df, pd.DataFrame):
            raise TypeError("The 'df' argument must be a pandas DataFrame.")
        if next_point and isinstance(next_point, dict) and len(next_point) != 3:
            raise TypeError("The 'next_point' must be None or a `dict` of (C, N, D).")
        # Ignore call
        if len(cc.database.df) < 3:
            self.logger.warning("Not enough samples in the database. Terminating plot")
            return

        # Overrides
        if fitted:
            if not all(hasattr(cc, param) for param in ["E", "A", "B", "alpha", "beta"]):
                self.logger.warning(
                    "The scaling law parameters has not been estimated. Plotting only raw data points."
                )
                fitted = False
        if not fitted:
            img_name = "raw_data_points"
        if simulation:
            img_name = f"simulation--{img_name}"
        img_filepath = os.path.join(self.project_dir, img_name + ".png")

        # set a shortcut
        self.cc = cc
        self._next_point = next_point
        self.cmap = plt.colormaps[cmap_name]
        df = self.cc.database.df

        # Pad the all points & marginalize the higher edge of the gradient
        margin = 0.20
        loss_min, loss_max = df.loss.min(), df.loss.max()
        loss_range = loss_max - loss_min
        if self._next_point:
            loss_min = min(loss_min, cc.L(self._next_point["N"], self._next_point["D"]))
        iso_losses = np.linspace(loss_min - loss_range * margin, loss_max + loss_range * margin, 32)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.tight_layout(pad=4.0, w_pad=3.0)

        # Execute helper methods by each subplot
        for i, (x, y) in enumerate([("C", "N"), ("C", "D"), ("N", "D")]):
            ax = axes[i]
            # Get the highest value to include for each axis
            x_max, y_max = [
                max(
                    self.cc.seed_ranges[k][1],
                    self.cc.database.df[k].max(),
                    self._next_point[k] if self._next_point else -float("inf"),
                )
                for k in [x, y]
            ]

            # Adjust subplot and get the bound
            xlim, ylim = self._adjust_subplot(ax, x, y, x_max, y_max)

            # Emphasize seeds
            self._shadow_seed_regime(ax, x, y)

            # Scatter plot of `df`
            self._scatter_data_points(ax, x, y, iso_losses)

            # Visualize the scaling law derived from `self.cc.fit`.
            if fitted:
                # Loss gradient
                self._plot_loss_gradient(ax, x, y, iso_losses, y_max)

                # Efficient frontier line on the gradient
                self._draw_efficient_frontier(ax, x, y, xlim, ylim)

                # show `next_point` along with a guideline if not None,
                self._show_next(ax, x, y)

            ax.legend(loc="upper left").set_zorder(1000)

        # Add a colorbar to the right
        self._add_colorbar_to_plot(fig, axes, iso_losses)

        # Finish
        plt.suptitle("Efficient frontier on iso-loss gradient" if fitted else "Distribution of seed models", y=0.925)
        plt.savefig(img_filepath)
        plt.show()
        plt.close()

        self.logger.info(f"Image saved to [u]{img_filepath}[/]")

    def LBFGS(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        C: np.ndarray | None = None,
        simulation: bool = False,
        img_name: str = "LBFGS",
    ) -> None:
        """
        Plots the results of L-BFGS optimization, including the loss history and prediction accuracy.
        This method visualizes the predicted values versus the true labels and the error distribution.

        **Example output**:
        ![](../examples/efficientcube-1e15_1e16/LBFGS.png)

        Args:
            y_pred (np.ndarray): Predicted values by the model.
            y_true (np.ndarray): True labels or values for comparison with predictions.
            C (np.ndarray | None, optional): Compute resources (FLOPs) associated with each prediction, if available.
            simulation (bool, optional): Whether the plot is for a simulation.
            img_name (str, optional): The name of the image file to save the plot as.
        """
        if simulation:
            img_name = f"simulation--{img_name}"

        margin = 0.10

        vrange = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
        self.logger.debug(f"Loss range: {vrange}")
        lims = vrange[0] - margin * (vrange[1] - vrange[0]), vrange[1] + margin * (vrange[1] - vrange[0])
        self.logger.debug(f"Loss range [u]for plotting[/]: {lims}")

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.tight_layout(pad=4.0, w_pad=3.0)
        axs[0].scatter(y_pred, y_true)
        axs[0].plot(lims, lims, ls=":", lw=1, c="black")
        axs[0].set_xlabel("Predictions")
        axs[0].set_ylabel("True labels")
        axs[0].set_xlim(*lims)
        axs[0].set_ylim(*lims)
        axs[0].set_title("Best predictions vs. True labels")

        err = y_true - y_pred
        if C is None:
            densities, bins, _ = axs[1].hist(err, density=True)
            x = np.linspace(0, err.max(), 100)
            # Check if we can fit the errors to a exponential distribution
            lmd = 1 / err.mean()
            if lmd > 0:
                # A quick test to see the goodness of fit
                from scipy.stats import ks_2samp

                cdf_data = np.cumsum(densities) / np.sum(densities)
                cdf_model = 1 - np.exp(-lmd * x)
                KS, p = ks_2samp(cdf_data, cdf_model)
                if p > 0.05:  # My apologies for using not just p-value but also the threshold.
                    self.logger.info(rf"Goodness-of-fit to Exp(λ={lmd:.2f}): {KS=}, {p=}")
                    axs[1].plot(x, lmd * np.exp(-lmd * x), ls=":", lw=1, c="black", label=rf"$λ={lmd:.2f}$")
                    axs[1].legend(loc="upper right")
            axs[1].set_ylim(0, densities.max() * 1.05)
            axs[1].set_xlabel(r"Error ($L-\hat{L}$)")
            axs[1].set_ylabel("Probability density")
            axs[1].set_title("Error distribution")
        else:
            axs[1].scatter(C, np.abs(err))
            axs[1].set_xlabel("Compute (FLOPs)")
            axs[1].set_ylabel("Absolute error")
            axs[1].set_xscale("log")
            axs[1].set_yscale("log")
            axs[1].set_title("Compute and absolute error")

        plt.suptitle("L-BFGS results")
        plt.savefig(os.path.join(self.project_dir, img_name + ".png"))
        plt.show()
        plt.close()

    def _plot_loss_gradient(self, ax, x, y, iso_losses, y_max):
        """Helper method to plot the loss gradient."""
        # / 1 for converting to float when int
        log_ymin = np.log10(self.cc.seed_ranges[y][0] / 1)
        log_ymax = np.log10(y_max / 1)
        log_ymin -= (log_ymax - log_ymin) * PADDING
        log_ymax += (log_ymax - log_ymin) * PADDING

        y_values = np.logspace(log_ymin, log_ymax, 1000, dtype=np.double)
        assert not np.isnan(y_values).sum(), (f"{100*np.isnan(y_values).mean()}% NaN:", y_values)
        for j, L in enumerate(iso_losses):
            if y == "N":
                N = y_values
                _ = self.cc.B / (L - (self.cc.E + np.exp(np.log(self.cc.A) - self.cc.alpha * np.log(N))))
                # `_` can have negatives where `L < E + A/(N^alpha)` or `L < E + B/(D^beta)`, which is invalid by design.
                # Here, we arbitrarily leave potentially negative values so as to replace as NaN with `np.power`
                with np.errstate(invalid="ignore"):
                    D = np.power(_, 1 / self.cc.beta)
            elif y == "D":
                D = y_values
                _ = self.cc.A / (L - (self.cc.E + np.exp(np.log(self.cc.B) - self.cc.beta * np.log(D))))
                # Save as for N
                with np.errstate(invalid="ignore"):
                    N = np.power(_, 1 / self.cc.alpha)
            else:
                raise NotImplementedError("Please implement it yourself.")

            C = 6 * N * D
            x_values = locals()[x]
            ax.plot(x_values, y_values, c=self.cmap(j / len(iso_losses)), zorder=1)

    def _shadow_seed_regime(self, ax, x, y, resolution: int = 100):
        """Helper method to fill the seed regime with gray."""
        if x == "C":
            c = np.logspace(*np.log10(self.cc.seed_ranges[x]), resolution)
            if y == "N":
                y_lower = np.sqrt(c / (6 * self.cc.seed_ranges.N_to_D[1]))
                y_upper = np.sqrt(c / (6 * self.cc.seed_ranges.N_to_D[0]))
            else:  # y == "D"
                y_lower = np.sqrt(c * self.cc.seed_ranges.N_to_D[0] / 6)
                y_upper = np.sqrt(c * self.cc.seed_ranges.N_to_D[1] / 6)
            x_values = c
        else:  # x in ["N", "D"]
            n = np.logspace(*np.log10(self.cc.seed_ranges[x]), resolution)
            y_lower = np.maximum(self.cc.seed_ranges["C"][0] / (6 * n), self.cc.seed_ranges.N_to_D[0] * n)
            y_upper = np.minimum(self.cc.seed_ranges["C"][1] / (6 * n), self.cc.seed_ranges.N_to_D[1] * n)
            x_values = n

        ax.fill_between(x_values, y_lower, y_upper, color="silver", alpha=0.5, zorder=0, label="Seed")

    def _adjust_subplot(self, ax, x, y, x_max, y_max):
        """Adjusts the subplot configurations and return the bound of values."""
        # Get a tuple of value range for each axis, letting the minimum of a seed range the lowest and padding the all data points
        limits_by_k = {}
        for k in [x, y]:
            # Converting to `float` in case of int dtype (`np.log` cannot intake astronomically large integers)
            max_value = {x: x_max, y: y_max}[k] / 1
            log_range = np.log(max_value) - np.log(self.cc.seed_ranges[k][0])
            limits_by_k[k] = (
                self.cc.seed_ranges[k][0] * np.exp(-log_range * PADDING),
                max_value * np.exp(log_range * PADDING),
            )

        # Apply the bounds
        xlim, ylim = limits_by_k[x], limits_by_k[y]
        if np.isnan(xlim).any():
            raise ValueError(f"NaN value(s) found for the x axis {x} bound: {xlim}")
        if np.isnan(ylim).any():
            raise ValueError(f"NaN value(s) found for the x axis {y} bound: {ylim}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xscale("log")
        ax.set_yscale("log")
        to_label = {"C": "FLOPs", "D": "Samples", "N": "Parameters"}
        ax.set_xlabel(f"{to_label[x]} (${x}$)")
        ax.set_ylabel(f"{to_label[y]} (${y}$)")

        return xlim, ylim

    def _scatter_data_points(self, ax, x, y, iso_losses):
        ax.scatter(
            self.cc.database.df[x],
            self.cc.database.df[y],
            c=self.cc.database.df.loss,
            vmin=iso_losses.min(),
            vmax=iso_losses.max(),
            lw=1,
            edgecolor="black",
            cmap=self.cmap,
            zorder=100,
        )

    def _show_next(self, ax, x, y):
        if self._next_point:
            ax.plot(
                self._next_point[x],
                self._next_point[y],
                "*",
                markersize=18,
                markeredgecolor="black",
                label="Next",
                zorder=1000,
            )
            kwargs = dict(ls=":", lw=1, c="black")
            # ax.axvline(self._next_point[x], ymax=self._next_point[y], **kwargs)
            # ax.axhline(self._next_point[y], xmax=self._next_point[x], **kwargs)
            ax.plot([self._next_point[x]] * 2, [0, self._next_point[y]], **kwargs)
            ax.plot([0, self._next_point[x]], [self._next_point[y]] * 2, **kwargs)

    def _draw_efficient_frontier(self, ax, x, y, xlim, ylim):
        if x == "C":
            c_lim = xlim
        elif y == "C":
            c_lim = ylim
        else:
            c_lim = (6 * xlim[0] * ylim[0], 6 * xlim[1] * ylim[1])

        allocations = np.array([self.cc.allocate_compute(C) for C in c_lim])
        efficient_frontier = {"C": c_lim, "N": allocations[:, 0], "D": allocations[:, 1]}
        ax.plot(efficient_frontier[x], efficient_frontier[y], ls=":", c="blue", zorder=10, label="Efficient frontier")

    def _add_colorbar_to_plot(self, fig, axes, iso_losses):
        ax = axes[-1]
        fig.subplots_adjust(right=0.875)
        cax = fig.add_axes(
            (0.9, ax.get_position().y0, 0.015, ax.get_position().y1 - ax.get_position().y0)
            # (left, bottom, width, height)
        )
        cbar = plt.cm.ScalarMappable(cmap=self.cmap)
        cbar.set_array([iso_losses])
        fig.colorbar(cbar, cax=cax)
        cax.set_ylabel("Loss", rotation=270, labelpad=16)
