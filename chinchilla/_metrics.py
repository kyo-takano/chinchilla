"""A few loss & weight functions you can use on demand."""
from __future__ import annotations  # PEP 604 backport

import numpy as np


def asymmetric_mae(y_true: np.ndarray, y_pred: np.ndarray, w: float = 1e1) -> np.ndarray:
    r"""Asymmetric Mean Absolute Error loss function."""
    error = y_true - y_pred
    error[error < 0] *= -w  # abs() and weight as a time
    return error


def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Huber loss function."""
    error = y_true - y_pred
    abs_err = np.abs(error)
    loss = np.where(abs_err < delta, np.power(error, 2) / 2, delta * (abs_err - delta / 2))
    return loss


def log_huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """The original loss function used in the Chinchilla paper"""
    y_true, y_pred = np.log(y_true), np.log(y_pred)
    return huber(y_true, y_pred, delta)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Absolute Error loss function."""
    return np.abs(y_true - y_pred)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean Squared Error loss function."""
    return np.power(y_true - y_pred, 2)
