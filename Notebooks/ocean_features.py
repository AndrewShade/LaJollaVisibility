"""
Feature logic shared by Feature Engineering (training) and Model Inference (serving).

Keeping it in one place means the model sees identically-computed features at training and
prediction time. All time shifts are by calendar day, so a missing day yields NaN instead of
silently reaching back to an older row.
"""
import numpy as np
import pandas as pd

CLASS_NAMES = ['Poor', 'Fair', 'Good', 'Excellent']  # the dive shop's own rating, ids 0-3
GO_CLASS = 3  # "Go" means the shop rated the day Excellent

# Columns that get 1-3 day lags
LAG_COLS = ['waveHs_mean', 'waveHs_max', 'wind_speed', 'wavePeakPSD_max',
            'rain_72h_weighted_mm', 'sstSeaSurfaceTemperature']


def day_shift(series, days):
    """Value from `days` calendar days earlier, aligned to the original index (NaN if that day is missing)."""
    return series.shift(days, freq='D').reindex(series.index)


def rain_weighted(rain_mm):
    """Decay-weighted rain: today + 0.6 * yesterday + 0.3 * two days ago (models runoff lag)."""
    return rain_mm + day_shift(rain_mm, 1) * 0.6 + day_shift(rain_mm, 2) * 0.3


def add_time_features(daily):
    """
    Adds seasonality, calendar-day lags, SST changes and swell trend to a daily frame
    indexed by normalized date (no label column needed).

    Args:
        daily (pd.DataFrame): Joined daily buoy/wind/rain/tide features.

    Returns:
        pd.DataFrame: Copy of `daily` with the extra columns.
    """
    df = daily.copy()

    doy = pd.to_datetime(df.index).dayofyear
    df['season_sine'] = np.sin(2 * np.pi * doy / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * doy / 365.25)

    for col in LAG_COLS:
        if col in df.columns:
            for n in (1, 2, 3):
                df[f'{col}_lag{n}'] = day_shift(df[col], n)

    if 'sstSeaSurfaceTemperature' in df.columns:
        df['sst_diff_24h'] = df['sstSeaSurfaceTemperature'] - day_shift(df['sstSeaSurfaceTemperature'], 1)
        df['sst_diff_48h'] = df['sstSeaSurfaceTemperature'] - day_shift(df['sstSeaSurfaceTemperature'], 2)

    if 'waveHs_mean' in df.columns:
        df['swell_trend_3d'] = df['waveHs_mean'] - day_shift(df['waveHs_mean'], 3)

    return df


def pier_daily_features(raw, asof_hour=20, min_readings=20):
    """
    Summarizes raw Scripps Pier readings (UTC index; turbidity/chl/cdom columns) per local
    (Pacific) day, using only readings before `asof_hour` so training matches what is known
    at an evening run. Days with fewer than `min_readings` good readings are NaN.

    Returns:
        pd.DataFrame: pier_* median/max features indexed by local date.
    """
    if raw.empty: return pd.DataFrame()
    local = raw.copy()
    local.index = local.index.tz_localize('UTC').tz_convert('America/Los_Angeles').tz_localize(None)
    local = local[local.index.hour < asof_hour]
    g = local.resample('D')
    counts = g.count()

    out = pd.DataFrame(index=counts.index)
    for name, stats in {'turbidity': ['median', 'max'], 'chl': ['median', 'max'], 'cdom': ['median']}.items():
        for st in stats:
            col = f"pier_{name}_{'med' if st == 'median' else 'max'}"
            out[col] = g[name].agg(st).where(counts[name] >= min_readings)
    return out


# ---- Evening (8 pm) model ----------------------------------------------------------------
# Everything is observed through the evening before the target day (no forecasts), so training
# and serving see the same kind of data. Yesterday's pier readings carry most of the signal;
# a walk-forward backtest showed larger feature sets just overfit ~500 labeled days.
PIER_FEATURES = ['pier_turbidity_med_lag1', 'pier_turbidity_max_lag1', 'pier_chl_med_lag1', 'pier_cdom_med_lag1']
OBSERVED_FEATURES = ['waveHs_mean_lag1', 'waveHs_max_lag1', 'rain_72h_weighted_mm_lag1', 'wind_speed_lag1',
                     'wavePeakPSD_max_lag1', 'sstSeaSurfaceTemperature_lag1']
EVENING_FEATURES = PIER_FEATURES + OBSERVED_FEATURES + ['season_sine', 'season_cos']
_LOG_FEATURES = PIER_FEATURES + ['rain_72h_weighted_mm_lag1']  # heavy-tailed


def evening_matrix(features):
    """Model input matrix from a feature frame: the evening columns, heavy-tailed ones log-scaled."""
    X = features[EVENING_FEATURES].astype(float).copy()
    for c in _LOG_FEATURES:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X


def build_evening_frame(obs, pier_daily, target_date):
    """
    Feature row for `target_date` built only from observations before it.

    Args:
        obs (pd.DataFrame): Observed daily buoy/wind/tide plus `rain_mm`.
        pier_daily (pd.DataFrame): Output of pier_daily_features.
        target_date (pd.Timestamp): Day being predicted.

    Returns:
        pd.DataFrame: One row (target_date) with the evening features, NaN where an input is missing.
    """
    target_date = pd.Timestamp(target_date).normalize()
    hist = obs[obs.index < target_date].copy()
    hist['rain_72h_weighted_mm'] = rain_weighted(hist['rain_mm'])
    frame = hist.reindex(hist.index.union([target_date]))  # empty row for the target day; lags fill from history
    frame = add_time_features(frame)
    if not pier_daily.empty:
        frame = frame.join(pier_daily.shift(1, freq='D').add_suffix('_lag1'), how='left')
    for c in PIER_FEATURES:
        if c not in frame.columns: frame[c] = np.nan
    return frame.loc[[target_date]]
