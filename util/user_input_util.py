from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from util.util import Model


def make_yes_no_validator():
    return lambda s: s.strip().lower() in ("y", "yes", "n", "no")


def make_yes_no_normalizer():
    return lambda s: s.strip().lower() in ("y", "yes")


def make_enum_validator(options):
    return lambda s, n=len(options): s.strip().isdigit() and 1 <= int(s.strip()) <= n


def make_enum_normalizer(options):
    return lambda s, opts=options: opts[int(s.strip()) - 1]


def make_int_range_validator(lo, hi):
    def _v(s: str) -> bool:
        try:
            v = int(s.strip())
        except ValueError:
            return False
        return lo <= v <= hi

    return _v


def int_normalizer():
    return lambda s: int(s.strip())


def make_float_range_validator(lo, hi):
    def _v(s: str) -> bool:
        try:
            v = float(s.strip())
        except ValueError:
            return False
        return lo <= v <= hi

    return _v


def float_normalizer():
    return lambda s: float(s.strip())


def make_csv_file_validator():
    def _v(s: str) -> bool:
        p = s.strip().strip('"').strip("'")
        return p.lower().endswith(".csv") and Path(p).exists()

    return _v


def csv_path_normalizer():
    return lambda s: str(Path(s.strip().strip('"').strip("'")).resolve())


VARIABLES = {
    "model": {
        "prompt": "Select model",
        "options": ["AR", "ARIMA", "LSTM", "ALL"],  # numbered menu
        "default": "AR",
        "validator": make_enum_validator([Model.AR, Model.ARIMA, Model.LSTM, Model.ALL]),
        "normalizer": make_enum_normalizer([Model.AR, Model.ARIMA, Model.LSTM, Model.ALL]),
        "hint": None,
    },
    "plot_time_series": {
        "prompt": "Plot time series? (Y/N)",
        "default": True,
        "validator": make_yes_no_validator(),
        "normalizer": make_yes_no_normalizer(),
        "hint": None,
    },
    "path_to_csv": {
        "prompt": "Path to CSV with historical prices",
        "default": "./data/HistoricalData_AMD.csv",
        "validator": make_csv_file_validator(),
        "normalizer": csv_path_normalizer(),
        "hint": "Must end with .csv and the file must already exist.",
    },
    "forecast_horizon": {
        "prompt": "Forecast horizon (1–20)",
        "default": 10,
        "validator": make_int_range_validator(1, 20),
        "normalizer": int_normalizer(),
        "hint": None,
    },
    "in_sample_forecast": {
        "prompt": "Forecast in-sample? (Y/N)",
        "default": True,
        "validator": make_yes_no_validator(),
        "normalizer": make_yes_no_normalizer(),
        "hint": "If yes, the forecast is for the last X elements of the series so that one can see the relation of the forecast to the actual data.",
    },
    "plot_acf_pacf": {
        "prompt": "Plot ACF & PACF? (Y/N)",
        "default": True,
        "validator": make_yes_no_validator(),
        "normalizer": make_yes_no_normalizer(),
        "hint": "Plot ACF & PACF of the data",
    },
    "plot_kde": {
        "prompt": "Plot KDE of original and differenced series? (Y/N)",
        "default": True,
        "validator": make_yes_no_validator(),
        "normalizer": make_yes_no_normalizer(),
        "hint": "Distribution plots for original and differenced series",
    },
    "use_bic": {
        "prompt": "Use BIC to select best model? (Y/N)",
        "default": True,
        "validator": make_yes_no_validator(),
        "normalizer": make_yes_no_normalizer(),
        "hint": "Alternative: expanding-window RMSE",
    },
    "max_d": {
        "prompt": "Max differencing level (d: 1–2)",
        "default": 1,
        "validator": make_int_range_validator(1, 2),
        "normalizer": int_normalizer(),
        "hint": "Max differencing to make the series stationary",
    },
    "max_p": {
        "prompt": "Max AR order (p: 1–10)",
        "default": 5,
        "validator": make_int_range_validator(1, 10),
        "normalizer": int_normalizer(),
        "hint": "Max number of lags for the AR component",
    },
    "max_q": {
        "prompt": "Max MA order (q: 1–10)",
        "default": 5,
        "validator": make_int_range_validator(1, 10),
        "normalizer": int_normalizer(),
        "hint": "Max number of previous errors for the MA part",
    },
    "window_size": {
        "prompt": "LSTM window size (3–10)",
        "default": 6,
        "validator": make_int_range_validator(3, 10),
        "normalizer": int_normalizer(),
        "hint": "Length of each window used for predictions",
    },
    "lstm_layer_size": {
        "prompt": "Select LSTM layer size",
        "options": [32, 64, 128],
        "default": 64,
        "validator": make_enum_validator([32, 64, 128]),
        "normalizer": make_enum_normalizer([32, 64, 128]),
        "hint": "Prefer the default",
    },
    "dense_layer_size": {
        "prompt": "Select dense layer size",
        "options": [16, 32, 64],
        "default": 32,
        "validator": make_enum_validator([16, 32, 64]),
        "normalizer": make_enum_normalizer([16, 32, 64]),
        "hint": "Prefer the default",
    },
    "epochs": {
        "prompt": "Epochs (10–100)",
        "default": 30,
        "validator": make_int_range_validator(10, 100),
        "normalizer": int_normalizer(),
        "hint": "Prefer the default to reduce compute",
    },
    "sampling_paths": {
        "prompt": "Sampling paths, more paths = longer computation (100–400)",
        "default": 300,
        "validator": make_int_range_validator(100, 400),
        "normalizer": int_normalizer(),
        "hint": "Number of different paths that get computed for each step to build confidence intervals",
    },
    "learning_rate": {
        "prompt": "Learning rate (0.0001–0.01)",
        "default": 0.001,
        "validator": make_float_range_validator(0.0001, 0.01),
        "normalizer": float_normalizer(),
        "hint": "Prefer the default",
    },
    "start_over": {
        "prompt": "Do you want to start over? (Y/N)",
        "default": False,
        "validator": make_yes_no_validator(),
        "normalizer": make_yes_no_normalizer(),
        "hint": None,
    },
}


def get_input(name):
    """
    Prompts for a single variable defined in VARIABLES, validates, and returns the normalized value.

    UX:
      - If the variable has 'options', a numbered menu is shown and the user enters a number.
      - Press ENTER to accept the default.
      - Enter '?' to show the hint (if available).
    """
    if name not in VARIABLES:
        raise KeyError(f"Unknown variable '{name}'")

    cfg = VARIABLES[name]
    prompt = cfg.get("prompt", name)
    default = cfg.get("default", None)
    options = cfg.get("options", None)
    validate: Callable[[str], bool] = cfg["validator"]
    normalize: Callable[[str], Any] = cfg["normalizer"]
    hint = cfg.get("hint")

    # Render menu (if enum/options)
    if options:
        for i, opt in enumerate(options, start=1):
            print(f"({i}) {opt}")

    # Compute default display and "raw" fallback
    if options and default is not None:
        # Default is the option value; show it as label, accept ENTER -> its index
        try:
            default_index = options.index(default) + 1
        except ValueError:
            default_index = None
        default_display = f"{default}" if default_index is not None else ""
        default_raw = str(default_index) if default_index is not None else None
    elif isinstance(default, bool):
        default_display = "Y" if default else "N"
        default_raw = "y" if default else "n"
    else:
        default_display = str(default) if default is not None else ""
        default_raw = str(default) if default is not None else None

    # Prompt loop
    while True:
        suffix = f" [{default_display}]" if default_display else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if raw == "?" and hint:
            print(f"? {hint}")
            # re-show options if any
            if options:
                for i, opt in enumerate(options, start=1):
                    print(f"{i}) {opt}")
            continue

        if raw == "" and default_raw is not None:
            raw = default_raw

        if validate(raw):
            return normalize(raw)

        print("Invalid input. Please try again.")
        if hint:
            print(f"  Hint: {hint}")
