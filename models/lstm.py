import datetime
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import BDay
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from util.user_input_util import get_input
from util.util import handle_unit_root


def df_to_windowed_df(df, window_size):
    """
    Convert it into a supervised learning problem by building test data.
    The data is transformed into (num_of_observations, window_size+2) output.
    The output has the form: Target Date|X(of length window_size|y
    """
    # Convert it into a supervised learning problem
    target_date = df.index[window_size]
    last_date = df.index[-1]

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = df.loc[:target_date].tail(window_size + 1)
        if len(df_subset) != window_size + 1:
            print(f'Error: Window of size {window_size} is too large for date {target_date}')
            return

        values = df_subset['Close/Last'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        # Get next date
        next_week = df.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, window_size):
        X[:, i]
        ret_df[f'X-{window_size - i}'] = X[:, i]

    ret_df['y'] = Y

    return ret_df


def windowed_df_to_date_X_y(windowed_df, window_size):
    """
    Splits each window into three distinc entries:
    - Date (length of windowed dataframe,)
    - X -> The lagged values of the target (length of windowed dataframe, window size)
    - y -> The target (length of windowed dataframe,)
    """
    df_as_np = windowed_df.to_numpy()

    dates = df_as_np[:, 0]
    middle_matrix = df_as_np[:, 1:-1]
    # the 1 at the end, because we do "univariate" analysis, e.g. we only consider 1 feature over time and not multiple
    X = middle_matrix.reshape((len(dates), window_size, 1))
    y = df_as_np[:, -1]

    return dates, X.astype(np.float32), y.astype(np.float32)


def sample_lstm_paths(model, last_window, window_size, horizon, n_paths):
    """
    Generates multiple forecast paths using a Monte Carlo simulation approach with an LSTM model.
    This function leverages stochasticity introduced by Dropout during inference to simulate
    uncertainty in predictions. By repeatedly sampling predictions with Dropout enabled, it
    produces a distribution of possible future outcomes, which can be used to estimate
    confidence intervals.
    """
    paths = np.zeros((n_paths, horizon), dtype=float)
    print("Building confidence intervals via sampling. This might take a while...")
    for k in range(n_paths):
        if k % 5 == 0:
            print(f"Sampling paths ... {k}/{n_paths}")
        w = last_window.copy()
        for t in range(horizon):
            yhat = model(w.reshape(1, window_size, 1), training=True).numpy().squeeze()
            paths[k, t] = yhat
            w = np.roll(w, -1, axis=0)
            w[-1, 0] = yhat

    print("Finished sampling.")
    return paths


def lstm(df, fc_horizon, in_sample_forecast):
    stationary_df, difference_level, _ = handle_unit_root(df, 1)
    var_window_size = get_input("window_size")
    var_lstm_layer_size = get_input("lstm_layer_size")
    var_dense_layer_size = get_input("dense_layer_size")
    var_epochs = get_input("epochs")
    var_learning_rate = get_input("learning_rate")
    val_breakpoint = 0.7 if in_sample_forecast else 0.8
    test_breakpoint = 0.8 if in_sample_forecast else 1

    windowed_df = df_to_windowed_df(stationary_df, var_window_size)

    dates, X, y = windowed_df_to_date_X_y(windowed_df, var_window_size)

    train_end = int(len(dates) * val_breakpoint)
    val_end = int(len(dates) * test_breakpoint)

    dates_train, X_train, y_train = dates[:train_end], X[:train_end], y[:train_end]
    dates_val, X_val, y_val = dates[train_end:val_end], X[train_end:val_end], y[train_end:val_end]

    model = Sequential([
        layers.Input((var_window_size, 1)),
        layers.SpatialDropout1D(0.2),
        layers.LSTM(var_lstm_layer_size, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(var_dense_layer_size, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(var_dense_layer_size, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer=Adam(learning_rate=var_learning_rate), metrics=['mean_absolute_error'])

    # Stop early if there is no improvement in val_loss
    early = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        restore_best_weights=True
    )
    weights_file_path = "best_lstm.weights.h5"
    # Established checkpoint to select model from best epoch
    ckpt = ModelCheckpoint(
        filepath=weights_file_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True
    )
    # Reduces learning rate if no improvement in val_loss
    reduce = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode="min"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=var_epochs,
        callbacks=[early, ckpt, reduce],
        verbose=1,
        shuffle=False
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))

    print(f"Best epoch was {best_epoch} with val_loss = {best_val_loss:.4f}")

    model.load_weights(weights_file_path)

    if os.path.exists(weights_file_path):
        os.remove(weights_file_path)

    last_date = df.index[-1]
    future_idx = pd.bdate_range(start=last_date + BDay(1), periods=fc_horizon)
    forecast_dates = df.index[-fc_horizon:] if in_sample_forecast else future_idx
    last_vals = deepcopy(
        stationary_df.iloc[-var_window_size:] if in_sample_forecast else stationary_df.iloc[-var_window_size:])

    # use Monte-Carlo with Dropouts to compute confidence intervals
    n_paths = get_input("sampling_paths")
    paths = sample_lstm_paths(model, last_vals.values, var_window_size, fc_horizon, n_paths=n_paths)

    mean = paths.mean(axis=0)
    lower_price = np.percentile(paths, 2.5, axis=0)
    upper_price = np.percentile(paths, 97.5, axis=0)
    last_seen_data_index = len(stationary_df[:len(stationary_df) - fc_horizon] if in_sample_forecast else stationary_df)

    # Transform log-differences back to original prices
    last_price = df[:last_seen_data_index + 1].iloc[-1, -1]
    mean_original = np.exp(last_price + mean.cumsum()) if difference_level > 0 else np.exp(mean)
    lower_price_original = np.exp(last_price + lower_price.cumsum()) if difference_level > 0 else np.exp(lower_price)
    upper_price_original = np.exp(last_price + upper_price.cumsum()) if difference_level > 0 else np.exp(upper_price)

    confidence_intervals = pd.DataFrame({"lower": lower_price_original, "upper": upper_price_original},
                                        index=forecast_dates)
    forecast_series = pd.Series(mean_original, index=forecast_dates, name="forecast")

    return forecast_series, confidence_intervals


if __name__ == '__main__':
    lstm()
