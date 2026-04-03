"""
Utility functions: Anomaly detection, Business Insights, Scenario Simulation.
"""
import pandas as pd
import numpy as np


def detect_anomalies(df, col='units_sold', window=14, threshold=2.5):
    """
    Z-score based anomaly detection on a time series.
    
    Returns DataFrame with z_score and is_anomaly columns.
    """
    df = df.copy()
    rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window, min_periods=1).std().replace(0, 1)
    
    df['z_score'] = (df[col] - rolling_mean) / rolling_std
    df['is_anomaly'] = df['z_score'].abs() > threshold
    df['anomaly_type'] = np.where(
        df['is_anomaly'],
        np.where(df['z_score'] > 0, 'Spike', 'Drop'),
        'Normal'
    )
    return df


def generate_business_insights(df):
    """
    Generate rich, data-driven business insights.
    """
    insights = []
    
    # ── Trend ──
    monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
    if len(monthly) >= 6:
        recent = monthly.iloc[-3:].mean()
        older = monthly.iloc[-6:-3].mean()
        if older > 0:
            change = (recent - older) / older * 100
            direction = "📈 growing" if change > 0 else "📉 declining"
            insights.append(f"Revenue is {direction} by {abs(change):.1f}% compared to the prior quarter.")
    
    # ── Top Region ──
    if 'region' in df.columns:
        reg_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
        top_reg = reg_rev.index[0]
        top_pct = reg_rev.iloc[0] / reg_rev.sum() * 100
        insights.append(f"🏆 {top_reg} leads with {top_pct:.1f}% of total revenue.")
    
    # ── Top Category ──
    if 'category' in df.columns:
        cat_units = df.groupby('category')['units_sold'].sum().sort_values(ascending=False)
        insights.append(f"🛒 {cat_units.index[0]} is the top-selling category ({cat_units.iloc[0]:,} units).")
    
    # ── Holiday Impact ──
    if 'holiday_flag' in df.columns:
        hol_avg = df[df['holiday_flag'] == 1]['units_sold'].mean()
        norm_avg = df[df['holiday_flag'] == 0]['units_sold'].mean()
        if norm_avg > 0 and not np.isnan(hol_avg):
            spike = (hol_avg - norm_avg) / norm_avg * 100
            if spike > 0:
                insights.append(f"🎉 Holiday periods show a {spike:.1f}% sales uplift vs normal days.")
    
    # ── Weekend Effect ──
    if 'weekend_flag' in df.columns:
        we_avg = df[df['weekend_flag'] == 1]['units_sold'].mean()
        wd_avg = df[df['weekend_flag'] == 0]['units_sold'].mean()
        if wd_avg > 0 and not np.isnan(we_avg):
            we_boost = (we_avg - wd_avg) / wd_avg * 100
            if we_boost > 0:
                insights.append(f"📅 Weekend sales are {we_boost:.1f}% higher than weekdays.")
    
    # ── Discount Impact ──
    if 'discount_percentage' in df.columns:
        high_disc = df[df['discount_percentage'] >= 20]['units_sold'].mean()
        low_disc = df[df['discount_percentage'] < 5]['units_sold'].mean()
        if low_disc > 0 and not np.isnan(high_disc):
            disc_lift = (high_disc - low_disc) / low_disc * 100
            if disc_lift > 0:
                insights.append(f"💰 High discounts (≥20%) drive {disc_lift:.1f}% more sales vs low/no discount periods.")
    
    # ── Stock Impact ──
    if 'stock_available' in df.columns:
        stockout_days = (df['stock_available'] == 0).sum()
        total_days = len(df)
        if total_days > 0:
            stockout_rate = stockout_days / total_days * 100
            if stockout_rate > 1:
                insights.append(f"⚠️ Stockout rate is {stockout_rate:.1f}%, causing potential revenue loss.")
    
    return insights


def generate_forecast(model, scaler_x, scaler_y, last_sequence, steps=30):
    """
    Multi-step rolling forecast using the trained LSTM model.
    
    Args:
        model: trained LSTM model
        scaler_x: feature scaler
        scaler_y: target scaler
        last_sequence: last (lookback, features) scaled sequence
        steps: number of days to forecast
    
    Returns:
        np.array of forecasted values (unscaled)
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        pred_scaled = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)
        predictions.append(pred_scaled[0, 0])
        
        # Roll the sequence forward: drop oldest, append prediction
        new_row = current_seq[-1].copy()
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1] = new_row
    
    # Inverse transform predictions
    preds_array = np.array(predictions).reshape(-1, 1)
    forecasted = scaler_y.inverse_transform(preds_array).ravel()
    
    return forecasted


def simulate_scenario(model, scaler_x, scaler_y, last_sequence, feature_cols,
                      discount_change=0, marketing_change=0, steps=30):
    """
    What-if scenario simulation.
    Modifies feature values and generates a new forecast.
    """
    modified_seq = last_sequence.copy()
    
    # Find feature indices
    disc_idx = None
    mkt_idx = None
    for i, col in enumerate(feature_cols):
        if 'discount' in col.lower():
            disc_idx = i
        if 'marketing' in col.lower():
            mkt_idx = i
    
    # Apply modifications to the sequence
    if disc_idx is not None and discount_change != 0:
        modified_seq[:, disc_idx] += discount_change / 100.0
        modified_seq[:, disc_idx] = np.clip(modified_seq[:, disc_idx], 0, 1)
    
    if mkt_idx is not None and marketing_change != 0:
        modified_seq[:, mkt_idx] += marketing_change / 1000.0
        modified_seq[:, mkt_idx] = np.clip(modified_seq[:, mkt_idx], 0, 1)
    
    return generate_forecast(model, scaler_x, scaler_y, modified_seq, steps)
