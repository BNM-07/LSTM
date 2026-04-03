"""
Preprocessing & Feature Engineering for LSTM/GRU Sales Forecasting.
Aggregates raw data, engineers leak-free temporal features, and prepares sequences.

Key design decisions:
  - ALL rolling/lag features use shift(1) to prevent data leakage
  - Only strong, meaningful features retained (lag, rolling mean/std)
  - Separate scalers for features (X) and target (y)
  - Scalers fitted ONLY on training data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """Load and parse the sales CSV."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df


def aggregate_daily(df, group_cols=['date', 'region']):
    """
    Aggregate raw product-level data to daily-region level
    for practical LSTM training.
    """
    agg = df.groupby(group_cols).agg(
        units_sold=('units_sold', 'sum'),
        revenue=('revenue', 'sum'),
        avg_price=('price_per_unit', 'mean'),
        avg_discount=('discount_percentage', 'mean'),
        total_marketing=('marketing_spend', 'sum'),
        avg_footfall=('customer_footfall', 'mean'),
        avg_conversion=('conversion_rate', 'mean'),
        holiday_flag=('holiday_flag', 'max'),
        weekend_flag=('weekend_flag', 'max'),
        avg_competitor_price=('competitor_price', 'mean'),
        avg_stock=('stock_available', 'mean'),
        avg_supply_delay=('supply_chain_delay_days', 'mean'),
        weather_index=('weather_index', 'first'),
        economic_index=('economic_index', 'first'),
    ).reset_index()

    return agg


def feature_engineering(df, target_col='units_sold', group_col='region'):
    """
    Apply leak-free feature engineering for LSTM/GRU.
    Processes per expected grouping (e.g., per region) independently to keep series distinct.
    """
    df = df.sort_values([group_col, 'date']).reset_index(drop=True)

    # ── Shifted target for leak-free feature computation ──
    shifted = df.groupby(group_col)[target_col].shift(1)

    # ── Lag features (naturally leak-free via shift) ──
    for lag in [1, 7, 14]:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)

    # ── Rolling statistics (leak-free: computed on shifted series) ──
    for window in [7, 14]:
        df[f'{target_col}_roll_mean_{window}'] = df.groupby(group_col)[target_col].apply(
            lambda x: x.shift(1).rolling(window=window).mean()
        ).reset_index(level=0, drop=True)
        df[f'{target_col}_roll_std_{window}'] = df.groupby(group_col)[target_col].apply(
            lambda x: x.shift(1).rolling(window=window).std()
        ).reset_index(level=0, drop=True)

    # ── Time features (cyclical — no leakage risk) ──
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # ── Binary flags (no leakage risk) ──
    if 'holiday_flag' in df.columns:
        df['holiday_flag'] = df['holiday_flag'].astype(int)
    if 'weekend_flag' in df.columns:
        df['weekend_flag'] = df['weekend_flag'].astype(int)

    # Replace inf values
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Drop NaN rows from lags/rolling (first ~14 rows)
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df, target_col='units_sold'):
    """
    Return the curated list of feature columns for model training.
    Only includes strong, meaningful features.
    """
    # Core features — lag and rolling (leak-free)
    core_features = [
        f'{target_col}_lag_1',
        f'{target_col}_lag_7',
        f'{target_col}_lag_14',
        f'{target_col}_roll_mean_7',
        f'{target_col}_roll_mean_14',
        f'{target_col}_roll_std_7',
        f'{target_col}_roll_std_14',
    ]

    # Cyclical time features
    time_features = [
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos',
    ]

    # Binary flags
    binary_features = []
    for col in ['holiday_flag', 'weekend_flag']:
        if col in df.columns:
            binary_features.append(col)

    # Context features (only if available and meaningful)
    context_features = []
    for col in ['avg_discount', 'avg_footfall', 'avg_stock']:
        if col in df.columns:
            context_features.append(col)

    all_features = core_features + time_features + binary_features + context_features
    # Only return columns that actually exist
    return [c for c in all_features if c in df.columns]


def prepare_sequences(df, lookback=30, target_col='units_sold', forecast_horizon=1, group_col='region'):
    """
    Build LSTM input sequences from the feature-engineered DataFrame.
    """
    feature_cols = get_feature_columns(df, target_col=target_col)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scalers globally
    features_scaled = scaler_x.fit_transform(df[feature_cols].values)
    target_scaled = scaler_y.fit_transform(df[[target_col]].values).ravel()

    # Attach back to df loosely to allow per-group extraction
    df_scaled = df[[group_col]].copy()
    for i, col in enumerate(feature_cols):
        df_scaled[col] = features_scaled[:, i]
    df_scaled['target'] = target_scaled

    X, y = [], []
    
    # Process each region independently
    for region, group in df_scaled.groupby(group_col, sort=False):
        group_feat = group[feature_cols].values
        group_targ = group['target'].values
        
        for i in range(lookback, len(group_feat) - forecast_horizon + 1):
            X.append(group_feat[i - lookback:i])
            y.append(group_targ[i + forecast_horizon - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y, scaler_x, scaler_y, feature_cols


def save_preprocessor(scaler_x, scaler_y, feature_cols, lookback, save_dir='models'):
    """Save all preprocessing artifacts."""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump({
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'lookback': lookback,
    }, os.path.join(save_dir, 'preprocessor.pkl'))


def load_preprocessor(save_dir='models'):
    """Load preprocessing artifacts."""
    return joblib.load(os.path.join(save_dir, 'preprocessor.pkl'))
