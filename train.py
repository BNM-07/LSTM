"""
Training Pipeline for LSTM / GRU Sales Forecasting.
End-to-end: Load → Aggregate → Engineer → Train → Evaluate → Save.

Usage:
    python train.py                  # Train LSTM (default)
    python train.py --model gru      # Train GRU
    python train.py --model both     # Train both models
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import pandas as pd
import numpy as np
import keras
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import (
    load_data, aggregate_daily, feature_engineering,
    prepare_sequences, save_preprocessor
)
from model_lstm import build_lstm_model, build_gru_model, get_callbacks


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    return {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2), 'mape': float(mape)}


def _prepare_data(data_path, lookback=30):
    """
    Load, aggregate, engineer features, and prepare train/test sequences.
    Returns all data artifacts needed for training.
    """
    print("\n📥 Loading data...")
    df_raw = load_data(data_path)
    print(f"   Raw data: {df_raw.shape[0]:,} rows")

    print("\n📊 Aggregating to daily-region level...")
    df_agg = aggregate_daily(df_raw, group_cols=['date', 'region'])
    print(f"   Aggregated data: {df_agg.shape[0]:,} rows")

    df_data = df_agg.copy()

    print("\n🔧 Engineering features...")
    df_feat = feature_engineering(df_data, target_col='units_sold')
    print(f"   Features: {len(df_feat.columns)} columns, {len(df_feat)} rows")

    # Time-based split (80/20) while preserving regions
    # Find cutoff date across all data to prevent overlapping
    unique_dates = df_feat['date'].sort_values().unique()
    cutoff_idx = int(len(unique_dates) * 0.8)
    cutoff_date = unique_dates[cutoff_idx]
    
    train_df = df_feat[df_feat['date'] < cutoff_date].copy()
    test_df = df_feat[df_feat['date'] >= cutoff_date].copy()
    print(f"\n✂️  Split on {str(cutoff_date)[:10]}: Train {len(train_df)} rows | Test {len(test_df)} rows")

    # Prepare sequences
    print(f"\n🧬 Preparing sequences (lookback={lookback})...")
    X_train, y_train, scaler_x, scaler_y, feature_cols = prepare_sequences(
        train_df, lookback=lookback, target_col='units_sold'
    )

    # Prepare test sequences using SAME scalers and feature columns
    test_features_scaled = scaler_x.transform(test_df[feature_cols].values)
    test_target_scaled = scaler_y.transform(test_df[['units_sold']].values).ravel()

    test_df_scaled = test_df[['region']].copy()
    for i, col in enumerate(feature_cols):
        test_df_scaled[col] = test_features_scaled[:, i]
    test_df_scaled['target'] = test_target_scaled

    X_test, y_test = [], []
    for region, group in test_df_scaled.groupby('region', sort=False):
        group_feat = group[feature_cols].values
        group_targ = group['target'].values
        
        for i in range(lookback, len(group_feat)):
            X_test.append(group_feat[i - lookback:i])
            y_test.append(group_targ[i])
            
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'scaler_x': scaler_x, 'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'test_df': test_df,
    }


def _train_single_model(model_type, data, save_dir='models', lookback=30,
                         epochs=50, batch_size=32):
    """
    Train a single model (LSTM or GRU) using pre-prepared data.
    """
    start_time = time.time()
    mt = model_type.upper()

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler_x = data['scaler_x']
    scaler_y = data['scaler_y']
    feature_cols = data['feature_cols']
    test_df = data['test_df']

    # Build model
    print(f"\n🏗️  Building {mt} model...")
    build_fn = build_lstm_model if model_type == 'lstm' else build_gru_model
    model = build_fn(
        input_shape=(lookback, X_train.shape[2]),
        units=128,
        num_layers=2,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    model.summary()

    # Train
    print(f"\n🚀 Training {mt} for up to {epochs} epochs...")
    callbacks = get_callbacks(save_dir=save_dir, patience=10, model_type=model_type)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print(f"\n📈 Evaluating {mt} on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    print(f"\n   📊 {mt} Test Metrics:")
    print(f"      RMSE:  {metrics['rmse']:,.2f}")
    print(f"      MAE:   {metrics['mae']:,.2f}")
    print(f"      MAPE:  {metrics['mape']:.2f}%")
    print(f"      R²:    {metrics['r2']:.4f}")

    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, f'sales_{model_type}_model.keras')
    model.save(model_path)

    # Save preprocessor (shared — only once is fine)
    save_preprocessor(scaler_x, scaler_y, feature_cols, lookback, save_dir)

    # Save model-specific metrics
    metrics['model_type'] = model_type
    metrics['train_samples'] = int(X_train.shape[0])
    metrics['test_samples'] = int(X_test.shape[0])
    metrics['lookback'] = lookback
    metrics['epochs_run'] = len(history.history['loss'])
    metrics['training_time_seconds'] = round(time.time() - start_time, 1)

    with open(os.path.join(save_dir, f'metrics_{model_type}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Also save as default metrics.json for backward compat
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(save_dir, f'training_history_{model_type}.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(hist_dict, f, indent=2)

    # Save actual vs predicted
    results_df = pd.DataFrame({
        'date': test_df['date'].iloc[lookback:lookback + len(y_true)].values,
        'actual': y_true,
        'predicted': y_pred,
    })
    results_df.to_csv(os.path.join(save_dir, f'test_predictions_{model_type}.csv'), index=False)
    results_df.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)

    elapsed = time.time() - start_time
    print(f"\n✅ {mt} training complete in {elapsed:.1f}s")
    print(f"   Artifacts saved to {save_dir}/")

    return model, history, metrics


def train_pipeline(data_path, model_type='lstm', save_dir='models',
                   lookback=30, epochs=50, batch_size=32):
    """
    Complete training pipeline. Supports LSTM, GRU, or both.
    """
    print("=" * 60)
    print(f"  Sales Forecasting — Training Pipeline ({model_type.upper()})")
    print("=" * 60)

    # Prepare data once
    data = _prepare_data(data_path, lookback=lookback)

    results = {}

    if model_type == 'both':
        for mt in ['lstm', 'gru']:
            print(f"\n{'='*60}")
            print(f"  Training {mt.upper()} model")
            print(f"{'='*60}")
            model, history, metrics = _train_single_model(
                mt, data, save_dir=save_dir, lookback=lookback,
                epochs=epochs, batch_size=batch_size
            )
            results[mt] = {'model': model, 'history': history, 'metrics': metrics}

        # Print comparison
        print("\n" + "=" * 60)
        print("  📊 Model Comparison")
        print("=" * 60)
        print(f"  {'Metric':<12} {'LSTM':>12} {'GRU':>12} {'Winner':>10}")
        print(f"  {'-'*46}")
        for metric_key in ['rmse', 'mae', 'mape', 'r2']:
            v_lstm = results['lstm']['metrics'][metric_key]
            v_gru = results['gru']['metrics'][metric_key]
            if metric_key == 'r2':
                winner = 'LSTM' if v_lstm > v_gru else 'GRU'
            else:
                winner = 'LSTM' if v_lstm < v_gru else 'GRU'
            print(f"  {metric_key.upper():<12} {v_lstm:>12.4f} {v_gru:>12.4f} {winner:>10}")

    else:
        model, history, metrics = _train_single_model(
            model_type, data, save_dir=save_dir, lookback=lookback,
            epochs=epochs, batch_size=batch_size
        )
        results[model_type] = {'model': model, 'history': history, 'metrics': metrics}

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sales forecasting models")
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'gru', 'both'],
                        help='Model type to train: lstm, gru, or both')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback window')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sales_data.csv')
    if os.path.exists(data_path):
        train_pipeline(
            data_path,
            model_type=args.model,
            save_dir='models',
            lookback=args.lookback,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print("❌ Data not found. Run data_generation.py first.")
