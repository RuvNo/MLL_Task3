import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import BDay
from scipy.stats import gaussian_kde
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf

from util.user_input_util import get_input
from util.util import handle_unit_root, perform_cross_validation


def plot_acf_pacf_basic(series, lags, title_prefix):
    """
    Plots the AR(p) models ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function).
    """
    if lags is None:
        lags = max(1, min(40, len(series) // 4))

    acf_vals = acf(series, nlags=lags, fft=False)
    pacf_vals = pacf(series, nlags=lags, method="ywm")

    # Confidence interval (approx 95%)
    conf = 1.96 / np.sqrt(len(series))

    fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=120)

    # --- ACF ---
    axes[0].bar(range(len(acf_vals)), acf_vals, width=0.3, color="skyblue", edgecolor="black")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axhline(conf, color="red", linestyle="--")
    axes[0].axhline(-conf, color="red", linestyle="--")
    axes[0].set_title(f"{title_prefix}: ACF")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Correlation")

    # --- PACF ---
    axes[1].bar(range(len(pacf_vals)), pacf_vals, width=0.3, color="lightgreen", edgecolor="black")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(conf, color="red", linestyle="--")
    axes[1].axhline(-conf, color="red", linestyle="--")
    axes[1].set_title(f"{title_prefix}: PACF")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Correlation")

    fig.tight_layout()

    plt.show()


def invert_forecasts(forecasts, df, difference_level):
    """
    Reverting the forecasts back to the original price scale.
    """
    if difference_level > 1:
        raise ValueError("difference_level must be <= 1")

    inverted_forecasts = [df.iloc[-1, 0]]
    if difference_level == 1:
        for forecast in forecasts:
            inverted_forecasts.append(inverted_forecasts[-1] + forecast)

        inverted_forecasts = pd.Series(inverted_forecasts).apply(np.exp)
        return inverted_forecasts[1:]
    else:
        return forecasts.apply(np.exp)


def select_lags(stationary_series, use_bic, max_lags, number_of_observations, forecast_horizon):
    """
    Selecting the best lag (p) for the AR(p) model.
    The lag can either be chosen via minimization of the models BIC (in-sample) or the predictions RMSE (out-of-sample).
    """
    max_p_eff = min(max_lags, max(10, number_of_observations // 3))

    results = []
    if use_bic:
        # Determine best model in-sample
        for p in range(0, max_p_eff):
            model = AutoReg(stationary_series, lags=p).fit()
            results.append((p, model.bic))

        res_df = pd.DataFrame(results, columns=["p", "BIC"])
        best_p = res_df.loc[res_df["BIC"].idxmin(), "p"]

    else:
        # Determine best model out-of-sample
        for p in range(0, max_p_eff):
            rmse = perform_cross_validation(stationary_series['Close/Last'], is_arima=False,
                                            forecast_horizon=forecast_horizon, p=p)
            results.append((p, rmse))

        res_df = pd.DataFrame(results, columns=["p", "RMSE"])
        best_p = res_df.loc[res_df["RMSE"].idxmin(), "p"]

    return best_p


def check_for_heteroscedasticity(observations_count, resid):
    arch_lags = min(12, max(2, observations_count // 10))
    _, lm_p, _, _ = het_arch(resid, nlags=arch_lags)

    return lm_p < 0.05


def plot_distribution(data, plot, title):
    flat_values = data.values.flatten()
    kde = gaussian_kde(flat_values)

    x = np.linspace(min(flat_values), max(flat_values), 200)  # evaluation grid
    y = kde(x)

    plot.plot(x, y, label="KDE")
    plot.hist(flat_values, bins=30, density=True, alpha=0.3, label="Histogram")
    plot.set_title(title)


def plot_distributions(original_data, stationary_data):
    """
    Plots KDE (Kernel Density Estimate) diagrams for the original data and the stationary (differenced) data.
    """
    fig, plots = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
    plot1, plot2 = plots

    plot_distribution(original_data, plot1, "Distribution of Stock Prices")
    plot_distribution(stationary_data, plot2, "Distribution of Differences")

    plt.legend()
    plt.show()


def ar(df, fc_horizon, in_sample_forecast):
    var_max_d = get_input("max_d")
    var_max_p = get_input("max_p")

    stationary_series, difference_level, adf_pval = handle_unit_root(df, var_max_d)

    if difference_level >= 1:
        var_plot_kde = get_input("plot_kde")
        if var_plot_kde:
            plot_distributions(df, stationary_series)

    # Split data
    train_data = stationary_series[:len(stationary_series) - fc_horizon] if in_sample_forecast else stationary_series
    test_data = stationary_series[-fc_horizon:] if in_sample_forecast else pd.Series([])

    # Use a clean RangeIndex during modeling to avoid freq/index warnings
    train_data_without_dates = train_data.reset_index(drop=True)

    var_plot_acf_pacf = get_input("plot_acf_pacf")
    if var_plot_acf_pacf:
        plot_acf_pacf_basic(train_data_without_dates, lags=min(40, len(train_data_without_dates) - 2), title_prefix="AR input")

    # Compute optimal number of autoregressive lags
    observations_count = len(stationary_series)
    var_use_bic_for_model_selection = get_input("use_bic")
    p = select_lags(stationary_series, var_use_bic_for_model_selection, var_max_p, observations_count, fc_horizon)

    model = AutoReg(train_data_without_dates, lags=p, trend="c").fit()
    resid = model.resid

    # Print metrics
    is_heteroscedastic = check_for_heteroscedasticity(observations_count, resid)
    print("Metrics:")
    print(model.summary())
    print(f"Chosen model: AR({p})")
    if is_heteroscedastic:
        print("Be careful - the data seems to be heteroscedastic, based on the residuals.")

    last_date = train_data.index[-1]
    future_idx = pd.bdate_range(start=last_date + BDay(1), periods=fc_horizon)

    start = len(train_data_without_dates)
    end = start + fc_horizon - 1
    pred = model.get_prediction(start=start, end=end)

    ci = pred.conf_int(alpha=0.05)
    ci.index = future_idx
    ci.columns = ["lower", "upper"]

    # Transform log-differences back to original prices for ci
    last_price = df[:len(train_data) + 1].iloc[-1, -1]
    lower_price = np.exp(last_price + ci["lower"].cumsum()) if difference_level > 0 else np.exp(ci["lower"])
    upper_price = np.exp(last_price + ci["upper"].cumsum()) if difference_level > 0 else np.exp(ci["upper"])

    confidence_intervals = pd.DataFrame({"lower": lower_price, "upper": upper_price}, index=future_idx)

    mean = pred.predicted_mean
    mean.index = future_idx
    forecast_original = invert_forecasts(mean, df, difference_level)
    forecast_series = pd.Series(forecast_original.values, index=test_data.index if in_sample_forecast else future_idx,
                                name="forecast")

    return forecast_series, confidence_intervals
