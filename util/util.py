from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


class Model(Enum):
    AR = "AR"
    ARIMA = "ARIMA"
    LSTM = "LSTM"
    ALL = "ALL"


def compute_rmse(data, forecasts):
    """
    Takes in two series of values and computes the Root Mean Squared Error (rmse) between them.
    """
    last_n_values = data.iloc[-len(forecasts):]
    last_n_values_series = pd.Series(last_n_values['Close/Last'], index=last_n_values.index)

    return np.sqrt(np.mean((last_n_values_series - forecasts) ** 2))


def plot_single_forecast(data, forecasts, confidence_intervals, in_sample_forecast):
    """
    Plots a single forecast in two diagrams (incl. confidence intervals):
    The first diagram shows the whole time series.
    The second diagram shows a zoomed in view so that one can see the forecast better.
    """
    fig, plots = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    plot1, plot2 = plots

    last_regular_value = data.iloc[-len(forecasts) - 1] if in_sample_forecast else data.iloc[-1]
    last_regular_index = data.index[-len(forecasts) - 1] if in_sample_forecast else data.index[-1]
    last_regular_entry_series = pd.Series([last_regular_value], index=[last_regular_index])

    forecasts = pd.concat([last_regular_entry_series, forecasts])

    plot_size = min(300, len(forecasts) ** 2 + 50)

    plot1.plot(data.index, data['Close/Last'])
    if in_sample_forecast:
        plot1.plot(data.index[-len(forecasts):], data['Close/Last'][-len(forecasts):])
    plot1.plot(forecasts.index, forecasts)
    if confidence_intervals is not None:
        plot1.fill_between(confidence_intervals.index, confidence_intervals["lower"], confidence_intervals["upper"],
                           alpha=0.2, label="95% CI")
    plot1.legend(["Training Data", "Test Data", "Forecast"] if in_sample_forecast else ["Training Data", "Forecast"])
    plot1.set_title("Whole Time Series")

    plot2.plot(data.index[-plot_size:], data['Close/Last'][-plot_size:])
    if in_sample_forecast:
        plot2.plot(data.index[-len(forecasts):], data['Close/Last'][-len(forecasts):])
    plot2.plot(forecasts.index, forecasts)
    if confidence_intervals is not None:
        plot2.fill_between(confidence_intervals.index, confidence_intervals["lower"], confidence_intervals["upper"],
                           alpha=0.2, label="95% CI")
    plot2.legend(["Training Data", "Test Data", "Forecast"] if in_sample_forecast else ["Training Data", "Forecast"])
    plot2.set_title(f"Last {plot_size} entries of Time Series")

    plt.tight_layout()
    plt.show(block=False)


def plot_multiple_forecasts(data, forecast_ar, forecast_arima, forecast_lstm, in_sample_forecast):
    """
    Plots three forecast in two diagrams (without confidence intervals):
    The first diagram shows the whole time series.
    The second diagram shows a zoomed in view so that one can see the forecast better.
    """
    fig, plots = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    plot1, plot2 = plots

    max_forecast_length = max(len(forecast_ar), len(forecast_arima), len(forecast_lstm))

    last_regular_value = data.iloc[-max_forecast_length - 1] if in_sample_forecast else data.iloc[-1]
    last_regular_index = data.index[-max_forecast_length - 1] if in_sample_forecast else data.index[-1]
    last_regular_entry_series = pd.Series([last_regular_value], index=[last_regular_index])

    forecast_ar = pd.concat([last_regular_entry_series, forecast_ar])
    forecast_arima = pd.concat([last_regular_entry_series, forecast_arima])
    forecast_lstm = pd.concat([last_regular_entry_series, forecast_lstm])

    plot_size = min(300, max_forecast_length ** 2 + 50)

    plot1.plot(data.index, data['Close/Last'])
    if in_sample_forecast:
        plot1.plot(data.index[-max_forecast_length:], data['Close/Last'][-max_forecast_length:])
    plot1.plot(forecast_ar.index, forecast_ar)
    plot1.plot(forecast_arima.index, forecast_arima)
    plot1.plot(forecast_lstm.index, forecast_lstm)
    plot1.legend(
        ["Training Data", "Test Data", "Forecast AR", "Forecast ARIMA", "Forecast LSTM"] if in_sample_forecast else [
            "Training Data", "Forecast AR", "Forecast ARIMA", "Forecast LSTM"])
    plot1.set_title("Whole Time Series")

    plot2.plot(data.index[-plot_size:], data['Close/Last'][-plot_size:])
    if in_sample_forecast:
        plot2.plot(data.index[-max_forecast_length:], data['Close/Last'][-max_forecast_length:])
    plot2.plot(forecast_ar.index, forecast_ar)
    plot2.plot(forecast_arima.index, forecast_arima)
    plot2.plot(forecast_lstm.index, forecast_lstm)
    plot2.legend(
        ["Training Data", "Test Data", "Forecast AR", "Forecast ARIMA", "Forecast LSTM"] if in_sample_forecast else [
            "Training Data", "Forecast AR", "Forecast ARIMA", "Forecast LSTM"])
    plot2.set_title(f"Last {plot_size} entries of Time Series")

    plt.tight_layout()
    plt.show(block=False)


def handle_unit_root(df, max_differences):
    """
    Makes the input series stationary, if possible within the max number of differences.
    To check for stationarity, the ADFuller-test ist employed.
    """
    y = df.copy()
    d = 0
    while True:
        pval = adfuller(y, autolag="AIC")[1]
        if pval < 0.05:
            return y, d, pval
        y = y.diff().dropna()

        d += 1
        if d > max_differences:
            raise ValueError("Series cannot be made stationary.")


def get_steps(total_entries, split_size, forecast_horizon):
    """
    Produces the step intervals for an expanding window estimation for model comparison.
    """
    result = []
    current = split_size
    while current < total_entries - forecast_horizon:
        result.append(current)
        current += split_size
    result.append(total_entries - forecast_horizon)
    return result


def forecast_arima(data, forecast_horizon, order, dates):
    """
    Forecasting ARIMA(p,d,q) and returning the forecasts.
    """
    model = ARIMA(data, order=order, dates=dates, freq='B', missing='drop')
    model_fit = model.fit()
    forecasts = model_fit.get_forecast(steps=forecast_horizon)

    return forecasts.predicted_mean


def forecast_ar(data, forecast_horizon, p):
    """
    Forecasting AR(p) and returning the forecasts.
    """
    model = AutoReg(data, lags=p, trend="c").fit()
    forecasts = model.forecast(steps=forecast_horizon)

    return forecasts


def perform_cross_validation(y, is_arima, forecast_horizon, p=0, order=(), step_size=400):
    """
    Performs expanding window cross validation to get the average Root Mean Squared Error (rmse) of a model.
    The cross validation is used in the grid search to find the optimal model parameters ('p' for AR and 'p', 'd' & 'q' for ARIMA)
    """
    dates = pd.date_range(start=y[0], periods=len(y), freq='B')

    steps = get_steps(len(y), step_size, forecast_horizon)

    order_rmses = []
    for step in steps:
        train_data_step = y[:step]
        validation_data = y[step:step + forecast_horizon]

        if is_arima:
            forecasts = forecast_arima(train_data_step, forecast_horizon, order, dates[:step])
        else:
            forecasts = forecast_ar(train_data_step, forecast_horizon, p)

        rmse = root_mean_squared_error(validation_data, forecasts)
        order_rmses.append(rmse)

    return np.mean(order_rmses)
