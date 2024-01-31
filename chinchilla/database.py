from __future__ import annotations  # PEP 604 backport

import os

import pandas as pd

from ._logger import get_logger


class Database:
    """
    Stores and manipulates scaling data in a Pandas DataFrame the default persistence to a CSV file.
    The Database class is used internally by a `Chinchilla` instance.

    If `project_dir` is provided, the DataFrame is initialized from the CSV file at that location.
    If the file does not exist or is empty, a new DataFrame is created. If `project_dir` is None,
    the DataFrame is kept in memory.

    **Default columns**:
    - `C` (float): Compute in FLOPs.
    - `N` (int): Number of parameters.
    - `D` (int): Number of data samples seen.
    - `loss` (float): Loss value (optional, use case dependent).

    Args:
        project_dir (str | None): Directory for the CSV file storage.
        columns (list[str]): Column names for the DataFrame.
        logger (Logger): Logger instance for database messages.
    """

    def __init__(
        self, project_dir: str | None = None, columns: list[str] = ["C", "N", "D", "loss"], log_level: int = 30
    ) -> None:
        """
        Initializes the Database instance.

        Args:
            project_dir (Optional[str]): The directory path to save the DataFrame as a CSV file.
                                         If None, the DataFrame will not be saved to disk.
            columns (List[str]): A list of column names for the DataFrame.
            log_level (int): The logging level for the logger instance.
        """
        self.logger = get_logger(log_level, name="chinchilla.database")
        self.project_dir = project_dir
        self.filepath = os.path.join(self.project_dir, "df.csv") if self.project_dir else None

        if self.filepath and (os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0):
            self.df = pd.read_csv(self.filepath)
            self.logger.debug("Data types:\n" + self.df.dtypes.T.to_string())
        else:
            self.logger.info(
                f"Initializing a database to be saved to: {self.filepath}"
                if self.filepath
                else "Initializing an in-memory database for Simulator."
            )
            # We define an empty database for when referencing the number of existing data points
            self.df = pd.DataFrame([], columns=columns)

    def append(self, **result: dict[str, float]) -> None:
        """
        Appends a new row of results to the DataFrame and updates the CSV file if `project_dir` is set.

        If 'C' is not provided in `result`, it is automatically calculated as $6ND$.
        All numerical values are rounded to the nearest integer to prevent scientific notation in large values.
        Additional columns provided by the user are appended to the DataFrame.

        Args:
            result (dict[str, float]): A dictionary containing the data to append. Must include 'N', 'D', and 'loss' keys.
                If 'C' is not provided in `result`, it is automatically calculated as $6ND$.
                All numerical values are rounded to the nearest integer to prevent losing precisions to scientific notation for large values.
                Additional columns provided by the user will be appended to the DataFrame without any conflicts.
        """
        if not result or {"N", "D", "loss"} - set(result.keys()):
            raise ValueError("The 'result' dictionary must contain 'N', 'D', and 'loss' keys.")

        if "C" not in result:
            result["C"] = 6 * result["N"] * result["D"]
        for k in ["C", "N", "D"]:
            result[k] = round(result[k])  # This helps prevent scientific notation of large values

        # Collect all columns added by the user
        cols_additional = [c for c in result.keys() if c not in self.df.columns]
        record = pd.DataFrame([result], columns=self.df.columns.tolist() + cols_additional)

        # self.df = pd.concat([self.df, record], ignore_index=True); raise NotImplementedError("Do not use this code.")
        # Concatenating to the empty database may cause a data type issue:
        # e.g., when calling `Chinchilla.predict_loss` at the end of `Chinchilla.fit` (fixed with `.astype(float)`)
        self.df = record if len(self.df) == 0 else pd.concat([self.df, record], ignore_index=True)

        if self.filepath:
            self.df.to_csv(self.filepath, index=False, float_format="%f")
