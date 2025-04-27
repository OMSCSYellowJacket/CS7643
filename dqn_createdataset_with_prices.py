import pandas as pd
import numpy as np
import torch
import random
import os
import glob
import warnings

eps = 1e-8  # needed for numerical stability
L = 756  # length of dataframe for date range of interest

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


def splitdata():
    """
    Split data into train, val, test.
    """
    # Delete CSV files in the folder
    for folder in [
        "model_data/test_data/",
        "model_data/val_data/",
        "model_data/train_data/",
    ]:
        os.makedirs(folder, exist_ok=True)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Deleting {len(csv_files)} files from {folder}")
        for file in csv_files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error removing file {file}: {e}")

    random.seed(45)

    # Get tickers in acl18 dataset and divide it in 50% test data, 50% validation data
    try:
        table_df = pd.read_csv("tick_lst/acl18.txt", sep="\t")
        table_lst = table_df.Symbol.to_list()
        valtick_lst = random.sample(table_lst, 44)
        testtick_lst = list()
    except FileNotFoundError:
        print("Error: tick_lst/acl18.txt not found. Cannot create val/test from it.")
        table_lst = []
        valtick_lst = []
        testtick_lst = list()

    # Save files into test and validation folders
    print("Processing acl18 tickers for val/test...")
    acl18_raw_path = "acl18/raw/"
    if not os.path.isdir(acl18_raw_path):
        print(f"Error: Directory not found: {acl18_raw_path}")
    else:
        for tick in table_lst:
            try:
                raw_df = pd.read_csv(f"{acl18_raw_path}{tick[1:]}.csv")
                # Ensure 'Date' column exists and handle potential errors
                if "Date" not in raw_df.columns:
                    print(f"Skipping {tick}: 'Date' column missing.")
                    continue
                # Convert Date column to datetime
                raw_df["Date"] = pd.to_datetime(raw_df["Date"])
                df = raw_df[
                    (raw_df.Date >= "2014-01-01") & (raw_df.Date <= "2016-12-31")
                ].copy()
                if df.empty:
                    continue

                df.dropna(inplace=True)
                Ldf = len(df)

                # Check if required columns exist and are numbers
                required_cols = ["Low", "Volume", "Close"]
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {tick}: Missing columns.")
                    continue
                if not all(
                    pd.api.types.is_numeric_dtype(df[col]) for col in required_cols
                ):
                    print(
                        f"Skipping {tick}: Columns 'Low', 'Volume', 'Close' must be numeric."
                    )
                    continue

                if df.Low.min() >= 1.0 and df.Volume.min() >= 500.0 and Ldf >= L:
                    if tick in valtick_lst:
                        df.to_csv(f"model_data/val_data/{tick[1:]}.csv")
                    else:
                        testtick_lst.append(tick)
                        df.to_csv(f"model_data/test_data/{tick[1:]}.csv")
                else:
                    pass
            except FileNotFoundError:
                print(f"Error: Raw file not found for {tick} in {acl18_raw_path}")
            except Exception as e:
                print(f"Error processing {tick} for val/test: {e}")

    # Save files into training folder
    print("Processing tick_data tickers for train...")
    traintick_lst = list()
    tick_data_path = "tick_data/"
    if not os.path.isdir(tick_data_path):
        print(f"Error: Directory not found: {tick_data_path}")
        exit()
    else:
        for fle in glob.glob(f"{tick_data_path}*.csv"):
            try:
                fsplt = "$" + os.path.basename(fle)
                tick = fsplt.rpartition(".")[0]

                if tick not in table_lst:
                    raw_df = pd.read_csv(fle)
                    if "Date" not in raw_df.columns:
                        print(f"Skipping {tick}: 'Date' column missing.")
                        continue
                    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
                    df = raw_df[
                        (raw_df["Date"] >= "2014-01-01")
                        & (raw_df["Date"] <= "2016-12-31")
                    ].copy()
                    if df.empty:
                        continue

                    df.dropna(inplace=True)
                    Ldf = len(df)

                    required_cols = ["Low", "Volume", "Close"]
                    if not all(col in df.columns for col in required_cols):
                        print(f"Skipping {tick}: Missing required columns.")
                        continue
                    if not all(
                        pd.api.types.is_numeric_dtype(df[col]) for col in required_cols
                    ):
                        print(
                            f"Skipping {tick}: Columns 'Low', 'Volume', 'Close' must be numeric."
                        )
                        continue

                    if df.Low.min() >= 1.0 and df.Volume.min() >= 500.0 and Ldf >= L:
                        traintick_lst.append(tick)
                        df.to_csv(f"model_data/train_data/{tick[1:]}.csv")
                    else:
                        pass
            except FileNotFoundError:
                print(f"Error: File not found: {fle}")
            except Exception as e:
                print(f"Error processing {fle} for train: {e}")
    print("leaving splitdata()")


def create_lag(arry, lag=5):
    """
    Create lagged features by concatenating shifted feature blocks.
    Assumes input array shape (N_tickers, N_days, N_features_orig + 1)
    Assumes N_features_orig = 16
    Output shape will be (N_tickers, N_days - lag, (N_features_orig * (lag + 1)) + 1)
                 or    (N_tickers, N_days - lag, 96 + 1) if N_features_orig = 16

    Parameters:
    array (numpy array): A numpy array used to create features. Label is assumed to be the last feature.
    lag (int, optional): Lag value for features. Default is 5.

    Returns:
    Numpy array with lag features. The last element in the features is the truth label.
    """
    n_tickers, n_days, n_cols = arry.shape
    # Number of original features before lagging
    n_features_orig = n_cols - 1

    if n_days <= lag:
        print(
            f"Error: Array length ({n_days}) is not greater than lag ({lag}). Cannot create lagged features. Returning empty array."
        )
        final_feature_dim = (n_features_orig * (lag + 1)) + 1
        return np.empty((n_tickers, 0, final_feature_dim), dtype=arry.dtype)

    # Initialize the lagged array with the current features + label for the first block
    # Start from day index 'lag' to have enough history
    # Shape: (N_tickers, N_days-lag, N_features_orig+1)
    lagged_ary = arry[:, lag:, :]

    # Concatenate lagged features (excluding the label)
    for t in range(1, lag + 1):
        # Get features from t steps ago, align with the current block
        # Shape: (N_tickers, N_days-lag, N_features_orig)
        lagged_features = arry[:, lag - t : -t, :-1]
        # Concatenate along feature axis (axis=2)
        # Insert lagged features before label of current block
        lagged_ary = np.concatenate((lagged_features, lagged_ary), axis=2)

    print(f"create_lag output shape: {lagged_ary.shape}")
    return lagged_ary


def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use Simple Moving Average for initial values if EMA causes issues with short data
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + eps)
    rsi = 100 - (100 / (1 + rs))
    rsi.fillna(50, inplace=True)
    return rsi


def calculate_bollinger_bands_percent(data, period=20, sigma=2):
    rolling_avg = data.rolling(period, min_periods=1).mean()
    rolling_std = data.rolling(period, min_periods=1).std()
    upper = rolling_avg + sigma * rolling_std
    lower = rolling_avg - sigma * rolling_std
    bbp = (data - lower) / (upper - lower + eps)
    bbp.fillna(0.5, inplace=True)
    return bbp


def calculate_stochastic_momentum_indicator(
    data, k_period=5, d_period=10, signal_period=5
):
    highest_high = data.rolling(window=k_period, min_periods=1).max()
    lowest_low = data.rolling(window=k_period, min_periods=1).min()
    delta = highest_high - lowest_low
    relative = data - (highest_high + lowest_low) / 2

    ema1_delta = delta.ewm(span=d_period, adjust=False).mean()
    ema2_delta = ema1_delta.ewm(span=d_period, adjust=False).mean()
    dema_delta = 2 * ema1_delta - ema2_delta

    ema1_relative = relative.ewm(span=d_period, adjust=False).mean()
    ema2_relative = ema1_relative.ewm(span=d_period, adjust=False).mean()
    dema_relative = 2 * ema1_relative - ema2_relative

    smi = 100 * (dema_relative / (dema_delta + eps))
    smi.fillna(0, inplace=True)

    signal = smi.ewm(span=signal_period, adjust=False).mean()
    signal.fillna(0, inplace=True)

    return smi, signal


def return_numpy_data_with_prices(flepth):
    """
    Calculates the features, truth label, and corresponding closing prices.

    Parameters:
    flepth (str): file path to location of csv data files

    Returns:
    tuple: (ticker_ary, prices_ary)
        ticker_ary (numpy.ndarray): Contains features and label (last column).
                                    Shape (N_tickers, N_days_after_dropna, N_features + 1)
        prices_ary (numpy.ndarray): Contains closing prices corresponding to ticker_ary rows.
                                    Shape (N_tickers, N_days_after_dropna)
    """
    all_feature_label_arrays = []
    all_price_arrays = []

    print(f"Processing files in: {flepth}")
    file_list = glob.glob(f"{flepth}/*.csv")
    if not file_list:
        print("Error: No CSV files found.")
        return np.empty((0, 0, 17)), np.empty((0, 0))

    for i_f, fle in enumerate(file_list):
        if (i_f + 1) % 50 == 0:
            print(
                f"  Processing file {i_f+1}/{len(file_list)}: {os.path.basename(fle)}"
            )
        try:
            ticker_df = pd.read_csv(fle)
            if ticker_df.empty or len(ticker_df) < 30:
                print(
                    f"  Skipping {os.path.basename(fle)}: Too few rows ({len(ticker_df)})."
                )
                continue
            required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            if not all(col in ticker_df.columns for col in required_cols):
                print(f"  Skipping {os.path.basename(fle)}: Missing required columns.")
                continue
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                ticker_df[col] = pd.to_numeric(ticker_df[col], errors="coerce")
            ticker_df.dropna(
                subset=["Open", "High", "Low", "Close", "Volume"], inplace=True
            )
            if ticker_df.empty:
                print(
                    f"  Skipping {os.path.basename(fle)}: Empty after dropping NA in OHLCV."
                )
                continue

            ticker_temp2 = ticker_df.copy()
            original_closes = ticker_temp2["Close"].copy()

            indicators = [
                "EMA20",
                "EMA100",
                "RSI",
                "MACD",
                "MACDSignal",
                "CCI",
                "BOLL",
                "MA5_MA10",
                "MTM6_MTM12",
                "ROC",
                "SMI",
            ]

            ticker_temp2["EMA12"] = (
                ticker_temp2["Close"].ewm(span=12, adjust=False).mean()
            )
            ticker_temp2["EMA26"] = (
                ticker_temp2["Close"].ewm(span=26, adjust=False).mean()
            )
            ticker_temp2["EMA20"] = (
                ticker_temp2["Close"].ewm(span=20, adjust=False).mean()
            )
            ticker_temp2["EMA100"] = (
                ticker_temp2["Close"].ewm(span=100, adjust=False).mean()
            )
            ticker_temp2["MACD"] = ticker_temp2["EMA12"] - ticker_temp2["EMA26"]
            ticker_temp2["MACDSignal"] = (
                ticker_temp2["MACD"].ewm(span=9, adjust=False).mean()
            )
            ticker_temp2["RSI"] = calculate_rsi(ticker_temp2["Close"])
            ticker_temp2["p_t"] = (
                ticker_temp2["High"] + ticker_temp2["Low"] + ticker_temp2["Close"]
            ) / 3
            sma_pt = ticker_temp2["p_t"].rolling(20, min_periods=1).mean()
            mad_pt = (
                (ticker_temp2["p_t"] - sma_pt).abs().rolling(20, min_periods=1).mean()
            )
            ticker_temp2["CCI"] = (ticker_temp2["p_t"] - sma_pt) / (
                0.015 * mad_pt + eps
            )
            ticker_temp2["BOLL"] = calculate_bollinger_bands_percent(
                ticker_temp2["Close"]
            )
            ticker_temp2["MA5"] = ticker_temp2["Close"].rolling(5, min_periods=1).mean()
            ticker_temp2["MA10"] = (
                ticker_temp2["Close"].rolling(10, min_periods=1).mean()
            )
            ticker_temp2["MA5_MA10"] = ticker_temp2["MA5"] / (
                ticker_temp2["MA10"] + eps
            )
            ticker_temp2["MTM6"] = ticker_temp2["Close"] / (
                ticker_temp2["Close"].shift(6) + eps
            )
            ticker_temp2["MTM12"] = ticker_temp2["Close"] / (
                ticker_temp2["Close"].shift(12) + eps
            )
            ticker_temp2["MTM6_MTM12"] = ticker_temp2["MTM6"] / (
                ticker_temp2["MTM12"] + eps
            )
            ticker_temp2["ROC"] = (
                100
                * (ticker_temp2["Close"] - ticker_temp2["Close"].shift(10))
                / (ticker_temp2["Close"].shift(10) + eps)
            )
            ticker_temp2["SMI"], _ = calculate_stochastic_momentum_indicator(
                ticker_temp2["Close"]
            )

            # Normalize Indicators
            valid_indicators = [
                ind for ind in indicators if ind in ticker_temp2.columns
            ]
            ticker_log = pd.DataFrame(index=ticker_temp2.index)
            for i in valid_indicators:
                ticker_log[i + "dif"] = (ticker_temp2[i] + eps) / (
                    ticker_temp2[i].shift(1) + eps
                )

            # Normalize OHLCV
            prev_close = ticker_temp2["Close"].shift(1) + eps
            ticker_log["Opendif"] = ticker_temp2["Open"] / prev_close
            ticker_log["Highdif"] = ticker_temp2["High"] / prev_close
            ticker_log["Lowdif"] = ticker_temp2["Low"] / prev_close
            ticker_log["Closedif"] = ticker_temp2["Close"] / prev_close
            ticker_log["Volumedif"] = ticker_temp2["Volume"] / (
                ticker_temp2["Volume"].shift(1) + eps
            )

            # Calculate Label
            ticker_log["Truth_lbl"] = ticker_temp2["Close"].shift(-1) / (
                ticker_temp2["Close"] + eps
            )

            # Align Prices and Drop NAs
            # Get index before dropping NAs from ticker_log
            original_index = ticker_log.index
            ticker_log.dropna(inplace=True)
            # Get index after dropping NAs
            valid_index = ticker_log.index
            if valid_index.empty:
                print(
                    f"  Skipping {os.path.basename(fle)}: No valid rows after indicator calculation/NA drop."
                )
                continue

            # Select prices corresponding to the valid rows
            aligned_prices = original_closes.loc[valid_index]

            # Convert to numpy arrays
            ticker_features_label_temp = ticker_log.values.astype(np.float32)
            ticker_prices_temp = aligned_prices.values.astype(np.float32)

            # Check for consistent lengths before adding
            if len(ticker_features_label_temp) == len(ticker_prices_temp):
                all_feature_label_arrays.append(ticker_features_label_temp)
                all_price_arrays.append(ticker_prices_temp)
            else:
                print(
                    f"  Skipping {os.path.basename(fle)}: Mismatch in length after NA drop ({len(ticker_features_label_temp)} vs {len(ticker_prices_temp)})."
                )

            # Memory management
            del ticker_df, ticker_temp2, ticker_log, original_closes, aligned_prices
            del ticker_features_label_temp, ticker_prices_temp

        except Exception as e:
            print(f"Error processing file {fle}: {e}")
            import traceback

            traceback.print_exc()

    # Check if any data was collected
    if not all_feature_label_arrays:
        print("Error: No ticker data successfully processed.")
        # Return empty arrays
        return np.empty((0, 0, 17)), np.empty((0, 0))

    # Stack the arrays
    # Use padding if necessary for consistent N_days, but ideally filter upstream in splitdata
    # For now, assume filter in splitdata ensures consistent length after dropna
    try:
        ticker_ary = np.stack(all_feature_label_arrays, axis=0)
        prices_ary = np.stack(all_price_arrays, axis=0)
    except ValueError as e:
        print(f"Error stacking arrays: {e}")
        exit()

    return ticker_ary, prices_ary


# Split the data into train, val, test directories
splitdata()

# Create the features AND prices for the train, val, test datasets
print("Processing train data...")
train_ary, prices_train = return_numpy_data_with_prices("model_data/train_data/")
print("Processing val data...")
val_ary, prices_val = return_numpy_data_with_prices("model_data/val_data/")
print("Processing test data...")
test_ary, prices_test = return_numpy_data_with_prices("model_data/test_data/")

# Check if data loading was successful
if train_ary.size == 0 or val_ary.size == 0 or test_ary.size == 0:
    print("Error: One or more datasets are empty after processing. Exiting.")
    exit()

# Apply lag features (Apply before saving)
print("Creating lag features...")
lag = 5
train_ary_lagged = create_lag(train_ary, lag=lag)
val_ary_lagged = create_lag(val_ary, lag=lag)
test_ary_lagged = create_lag(test_ary, lag=lag)

# Adjust price arrays to match the potentially reduced N_days after lagging
# Assuming create_lag output shape is (N_tickers, N_days - lag, N_features_lagged + 1)
# Then we need to slice the prices to match
n_days_train_orig = prices_train.shape[1]
n_days_val_orig = prices_val.shape[1]
n_days_test_orig = prices_test.shape[1]

# Check if create_lag actually changed the time dimension length
days_train_lagged = train_ary_lagged.shape[1]
days_val_lagged = val_ary_lagged.shape[1]
days_test_lagged = test_ary_lagged.shape[1]

if days_train_lagged < n_days_train_orig:
    print(
        f"Adjusting price array length due to lag (assuming first {n_days_train_orig - days_train_lagged} days dropped)."
    )
    prices_train_final = prices_train[:, (n_days_train_orig - days_train_lagged) :]
    prices_val_final = prices_val[:, (n_days_val_orig - days_val_lagged) :]
    prices_test_final = prices_test[:, (n_days_test_orig - days_test_lagged) :]
    # Verify shapes
    assert prices_train_final.shape[1] == train_ary_lagged.shape[1]
    assert prices_val_final.shape[1] == val_ary_lagged.shape[1]
    assert prices_test_final.shape[1] == test_ary_lagged.shape[1]
else:
    print("Error: Lag creation did not reduce the number of days.")
    exit()


print("training, test, and val arrays/prices are created")
print("----------\n")

print("Array shape is (N_tickers, N_trading_days_effective, N_features_lagged + label)")
# (16 metrics * (5 lag + 1 current) = 96)
# So the last dimension should be 97
expected_feature_dim = 97
if train_ary_lagged.shape[2] != expected_feature_dim:
    print(
        f"Error: Expected {expected_feature_dim} columns (96 features + 1 label) after lag, but got {train_ary_lagged.shape[2]}."
    )
    exit()

print(f"Number of lagged features + label: {train_ary_lagged.shape[2]}")
print("Label is array[:,:,-1]")
print("---------\n")

print("Lagged train data shape:", train_ary_lagged.shape)
print("Lagged test data shape:", test_ary_lagged.shape)
print("Lagged val data shape:", val_ary_lagged.shape)
print("Final train prices shape:", prices_train_final.shape)
print("Final test prices shape:", prices_test_final.shape)
print("Final val prices shape:", prices_val_final.shape)

os.makedirs("DQN", exist_ok=True)

np.save("DQN/train_ary.npy", train_ary_lagged)
np.save("DQN/val_ary.npy", val_ary_lagged)
np.save("DQN/test_ary.npy", test_ary_lagged)
np.save("DQN/prices_train.npy", prices_train_final)
np.save("DQN/prices_val.npy", prices_val_final)
np.save("DQN/prices_test.npy", prices_test_final)

print("Saved lagged feature/label arrays and corresponding price arrays.")


try:
    train_ary_loaded = np.load("DQN/train_ary.npy")
    prices_train_loaded = np.load("DQN/prices_train.npy")

    print("Loaded train_ary shape:", train_ary_loaded.shape)
    print("Loaded prices_train shape:", prices_train_loaded.shape)

    # Preview the data
    if train_ary_loaded.shape[0] > 0 and train_ary_loaded.shape[1] > 3:
        print("First ticker, first 3 days, first 5 features:")
        print(train_ary_loaded[0, :3, :5])
        print("First ticker, first 3 labels:")
        print(train_ary_loaded[0, :3, -1])
        print("First ticker, first 3 closing prices:")
        print(prices_train_loaded[0, :3])
    else:
        print("Loaded arrays are too small to preview.")

except FileNotFoundError:
    print("Error: Saved files not found.")
