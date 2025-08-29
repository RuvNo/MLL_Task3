import itertools

import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import BDay
from statsmodels.tsa.arima.model import ARIMA

from util.user_input_util import get_input
from util.util import handle_unit_root, perform_cross_validation


def get_bic(data, dates, order):
    """
    Computes the BIC of an ARIMA(p,d,q) model to evaluate its performance.
    """
    model = ARIMA(data, order=order, dates=dates, freq='B', missing='drop')
    model_fit = model.fit()

    return model_fit.bic


def grid_search(train_data, dates, use_bic, max_d, max_p, max_q, fc_horizon):
    """
    Employs grid-search to find the optimal values for the auto-regressive (p) and moving-average (q) component.
    The grid-search is either via the BIC (in-sample) or via the RMSE of the forecast (out-of-sample).
    To save computational costs, the difference level is not determined by grid-search but rather via the ADFuller test.
    """
    p = range(0, max_p)
    _, difference_level, _ = handle_unit_root(pd.Series(train_data), max_d)
    d = [difference_level]
    q = range(0, max_q)

    pdq = list(itertools.product(p, d, q))
    best_rmse, best_bic, best_order = float("inf"), float("inf"), None

    for order in pdq:
        try:
            if use_bic:
                # Determine best model in-sample
                bic = get_bic(train_data, dates=dates, order=order)

                if bic < best_bic:
                    best_bic, best_order = bic, order
            else:
                # Determine best model out-of-sample
                rmse = perform_cross_validation(train_data, is_arima=True, forecast_horizon=fc_horizon,
                                                order=order)

                if rmse < best_rmse:
                    best_rmse, best_order = rmse, order

        except Exception as e:
            print(e)
            continue

    return best_order, best_bic, best_rmse


def arima(df, fc_horizon, in_sample_forecast):
    var_max_d = get_input("max_d")
    var_max_p = get_input("max_p")
    var_max_q = get_input("max_q")

    # Split data
    train_data = df[:len(df) - fc_horizon] if in_sample_forecast else df
    test_data = df[-fc_horizon:] if in_sample_forecast else pd.Series([])

    train_data = train_data['Close/Last']  # Data must be 1-D
    dates = pd.date_range(start=train_data.index[0], periods=len(train_data), freq='B')
    train_data_without_dates = train_data.to_numpy()

    var_use_bic_for_model_selection = get_input("use_bic")
    best_params, best_bic, best_rmse = grid_search(train_data_without_dates, dates=dates,
                                                   use_bic=var_use_bic_for_model_selection, max_d=var_max_d,
                                                   max_p=var_max_p, max_q=var_max_q, fc_horizon=fc_horizon)

    model = ARIMA(train_data_without_dates, order=best_params, dates=dates, freq='B', missing='drop')

    # Print metrics
    model_fit = model.fit()
    print("Metrics:")
    print(model_fit.summary())
    print(
        f"Chosen ARIMA parameters: {best_params}. {f'BIC: {best_bic:.4f}' if var_use_bic_for_model_selection else f'RMSE: {best_rmse:.4f}'}")

    forecasts = model_fit.get_forecast(steps=fc_horizon)

    last_date = train_data.index[-1]
    future_idx = pd.bdate_range(start=last_date + BDay(1), periods=fc_horizon)

    ci = forecasts.conf_int(alpha=0.05)
    ci = pd.DataFrame(ci, index=future_idx, columns=["lower", "upper"])
    lower_price = np.exp(ci["lower"])
    upper_price = np.exp(ci["upper"])
    confidence_intervals = pd.DataFrame({"lower": lower_price, "upper": upper_price}, index=future_idx)

    # Transform logs back to original prices
    predictions_logs = forecasts.predicted_mean
    predictions = np.exp(predictions_logs)

    forecast_series = pd.Series(predictions, index=test_data.index if in_sample_forecast else future_idx,
                                name="forecast")

    return forecast_series, confidence_intervals
