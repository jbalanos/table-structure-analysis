from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd

def continuity_metrics(
    df: pd.DataFrame,
    *,
    window: int = 20,                               # lookback W (previous rows only)
    masks: Optional[Dict[str, pd.DataFrame]] = None # optional: {'null','numeric','date'} → bool DFs aligned to df
) -> pd.DataFrame:
    """
    Compute C1 continuity metrics for a sheet-like DataFrame using ONLY previous rows as baseline.

    Returns a DataFrame with columns:
      - tcs_type_conformity         (float in [0,1]; 0.5 neutral when no history)
      - fcs_fill_continuity         (float in [0,1]; 1.0 when both current & baseline have no filled cells)
      - nbc_numeric_band_continuity (float in [0,1]; 1.0 when both current & baseline have no numeric cells)
    """

    R, C = df.shape
    idx = df.index

    # ---------------------------------------------------------------------
    # 1) Build or accept boolean cell-type masks
    #    null: blank/empty
    #    numeric: numeric-like (naive; supply your own via `masks` for locale-aware parsing)
    #    date: date-like
    #    text: remainder (non-null and not numeric/date)
    # ---------------------------------------------------------------------
    if masks is not None:
        null_mask_df    = masks.get('null')
        numeric_mask_df = masks.get('numeric')
        date_mask_df    = masks.get('date')
        if null_mask_df is None or numeric_mask_df is None or date_mask_df is None:
            raise ValueError("masks must contain boolean DataFrames for keys: 'null', 'numeric', 'date'")
    else:
        # naive inference (good enough for continuity patterns; you can pass better masks)
        sdf = df.astype("string").fillna("")
        stripped = sdf.apply(lambda col: col.str.strip())
        null_mask_df = (stripped == "")
        numeric_mask_df = df.apply(lambda col: pd.to_numeric(col, errors="coerce").notna())
        date_mask_df = df.apply(lambda col: pd.to_datetime(col, errors="coerce").notna())

    text_mask_df = (~null_mask_df) & (~numeric_mask_df) & (~date_mask_df)

    # Encode cell type per cell as small integers:
    # 0=blank, 1=text, 2=numeric, 3=date
    type_code_df = (
        null_mask_df.astype(int)   * 0 +
        text_mask_df.astype(int)   * 1 +
        numeric_mask_df.astype(int)* 2 +
        date_mask_df.astype(int)   * 3
    )

    # ---------------------------------------------------------------------
    # 2) Rolling “baseline” over previous W rows (NEVER includes current row)
    #    - For type: per-column rolling *mode*
    #    - For boolean masks (filled/numeric): per-column rolling *majority* (>=50%)
    # ---------------------------------------------------------------------
    modal_type_df = _rolling_modal_type(type_code_df, window)   # per column
    nn_mask_df    = ~null_mask_df
    modal_nn_df   = _rolling_majority_prev(nn_mask_df, window)  # filled baseline
    modal_num_df  = _rolling_majority_prev(numeric_mask_df, window)

    # ---------------------------------------------------------------------
    # 3) C1 metrics
    #    TCS: share of columns where current type == rolling modal type
    #    FCS: Jaccard(current filled, baseline filled)
    #    NBC: Jaccard(current numeric, baseline numeric)
    # ---------------------------------------------------------------------
    # TCS
    valid = modal_type_df.notna().to_numpy()
    matches = (type_code_df.to_numpy() == np.nan_to_num(modal_type_df.to_numpy(), nan=-1))
    denom = valid.sum(axis=1)
    tcs = np.divide((matches & valid).sum(axis=1), denom, out=np.full(R, 0.5), where=denom > 0)

    # FCS (filled continuity)
    inter = (nn_mask_df & modal_nn_df).sum(axis=1).to_numpy()
    union = (nn_mask_df | modal_nn_df).sum(axis=1).to_numpy()
    fcs = np.divide(inter, union, out=np.ones(R), where=union > 0)  # when both empty → continuity=1.0

    # NBC (numeric continuity)
    inter_n = (numeric_mask_df & modal_num_df).sum(axis=1).to_numpy()
    union_n = (numeric_mask_df | modal_num_df).sum(axis=1).to_numpy()
    nbc = np.divide(inter_n, union_n, out=np.ones(R), where=union_n > 0)

    return pd.DataFrame({
        "tcs_type_conformity":         tcs.astype(float),
        "fcs_fill_continuity":         fcs.astype(float),
        "nbc_numeric_band_continuity": nbc.astype(float),
    }, index=idx)


# =======================
# ------- Helpers -------
# =======================

def _rolling_modal_type(type_code_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Per-column rolling *mode* over previous W rows (shifted by 1 so it excludes the current row).
    Ties are broken deterministically by the smallest code (np.unique ordering).
    """
    cols = []
    for col in type_code_df.columns:
        s = type_code_df[col].astype('float')  # allow NaN in windows with no history
        cols.append(_rolling_mode_prev(s, window))
    out = pd.concat(cols, axis=1)
    out.columns = type_code_df.columns
    return out

def _rolling_mode_prev(series: pd.Series, win: int) -> pd.Series:
    """
    Rolling mode over the last W rows, **excluding** the current row.
    For row i, uses series[i-W : i].
    """
    def _mode(window_series: pd.Series) -> float:
        vals = window_series.dropna().to_numpy()
        if vals.size == 0:
            return np.nan
        u, cnt = np.unique(vals, return_counts=True)
        return float(u[np.argmax(cnt)])  # tie → smallest u
    return series.rolling(win, min_periods=1).apply(_mode, raw=False).shift(1)

def _rolling_majority_prev(mask_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Per-column rolling *majority* (>= 50% True) over previous W rows (excludes current row).
    """
    trues = mask_df.rolling(window, min_periods=1).sum().shift(1)
    cnts  = mask_df.rolling(window, min_periods=1).count().shift(1)
    return (trues >= (cnts / 2))
