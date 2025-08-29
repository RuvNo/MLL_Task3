import sys
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from models.ar import ar
from models.arima import arima
from models.lstm import lstm
from util.preprocess_util import preprocess, take_log
from util.user_input_util import get_input
from util.util import Model, plot_single_forecast, plot_multiple_forecasts, compute_rmse


def main():
    # Filter warnings that are not relevant for the user
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    var_file_path = get_input("path_to_csv")
    data = pd.read_csv(var_file_path)

    if len(data) < 1500:
        raise ValueError("Need at least ~1500 observations.")

    data = preprocess(data)

    var_plot_time_series = get_input("plot_time_series")
    if var_plot_time_series:
        plt.plot(data.index, data['Close/Last'])
        plt.legend(['Close prices'])
        plt.title(f"Close price for selected stock")
        plt.show(block=False)

    var_model = get_input("model")
    var_forecast_horizon = get_input("forecast_horizon")
    var_in_sample_forecast = get_input("in_sample_forecast")

    forecasts, confidence_intervals = None, None

    match var_model:
        case Model.AR:
            print("# ---------------------------")
            print("# Chosen model is AR")
            print("# ---------------------------")
            log_data = take_log(data.copy())
            forecasts, confidence_intervals = ar(log_data, var_forecast_horizon, var_in_sample_forecast)
        case Model.ARIMA:
            print("# ---------------------------")
            print("# Chosen model is ARIMA")
            print("# ---------------------------")
            log_data = take_log(data.copy())
            forecasts, confidence_intervals = arima(log_data, var_forecast_horizon, var_in_sample_forecast)
        case Model.LSTM:
            print("# ---------------------------")
            print("# Chosen model is LSTM")
            print("# ---------------------------")
            log_data = take_log(data.copy())
            forecasts, confidence_intervals = lstm(log_data, var_forecast_horizon, var_in_sample_forecast)
        case Model.ALL:
            log_data = take_log(data.copy())
            print("# ---------------------------")
            print("# Starting with AR")
            print("# ---------------------------")
            forecast_ar, _ = ar(log_data.copy(), var_forecast_horizon, var_in_sample_forecast)
            print("# ---------------------------")
            print("# Continuing with ARIMA")
            print("# ---------------------------")
            forecast_arima, _ = arima(log_data.copy(), var_forecast_horizon, var_in_sample_forecast)
            print("# ---------------------------")
            print("# Finishing with LSTM")
            print("# ---------------------------")
            forecasts_lstm, _ = lstm(log_data.copy(), var_forecast_horizon, var_in_sample_forecast)

            if var_in_sample_forecast:
                rmse_ar = compute_rmse(data, forecast_ar)
                rmse_arima = compute_rmse(data, forecast_arima)
                rmse_lstm = compute_rmse(data, forecasts_lstm)
                print(f"RMSE for AR: {rmse_ar:.4f}")
                print(f"RMSE for ARIMA: {rmse_arima:.4f}")
                print(f"RMSE for LSTM: {rmse_lstm:.4f}")

            plot_multiple_forecasts(data, forecast_ar, forecast_arima, forecasts_lstm, var_in_sample_forecast)
            return
        case _:
            raise ValueError(f"Unknown model: {var_model}")

    if var_in_sample_forecast:
        rmse = compute_rmse(data, forecasts)
        print(f"RMSE: {rmse:.4f}")

    plot_single_forecast(data, forecasts, confidence_intervals, var_in_sample_forecast)


def run_with_retry():
    while True:
        try:
            main()
            start_over = get_input("start_over")
            if start_over:
                plt.close("all")
                print("# ------------")
                print("Okay — restarting the process now.")
                print("# ------------")
            else:
                break
        except KeyboardInterrupt:
            print("\n# ---------------------------------------------------")
            print("Shutting down gracefully...")
            print("If you want to go again, just run 'python main.py'. :)")
            print("# ---------------------------------------------------")
            sys.exit(0)
        except Exception as e:
            print(f"Sorry, it seems like something went wrong.")
            print(f"An error occurred: {e}. Retrying...")


if __name__ == "__main__":
    print("# ---------------------------------------------------------------------------------------")
    print("Welcome to the stock price forecasting tool 'The one and only crystal ball'.")
    print("You will be guided throughout the setup.")
    print("If you do not want to use any custom settings, just press 'ENTER' to use the default.")
    print("If you do not understand one of the selections, try pressing '?' for a hint.")
    print("# ---------------------------------------------------------------------------------------")
    time.sleep(2)
    run_with_retry()
