from __future__ import annotations
import re
from typing import Tuple, Iterable, List
import numpy as np
import pandas as pd


def _infer_locale_from_strings(strings: Iterable[str]) -> Tuple[str, str]:
    """
    Infer locale-specific number formatting from sample strings.

    Args:
        strings: Iterable of string samples to analyze

    Returns:
        Tuple of (thousands_separator, decimal_separator)
        Returns (',', '.') for EN format or ('.', ',') for EU format
    """
    _NUM_EN_COMMA_DOT = re.compile(r"\b\d{1,3}(,\d{3})+(\.\d+)?\b")
    _NUM_EU_DOT_COMMA = re.compile(r"\b\d{1,3}(\.\d{3})+(,\d+)?\b")
    lst = list(strings)
    en_hits = sum(bool(_NUM_EN_COMMA_DOT.search(x)) for x in lst)
    eu_hits = sum(bool(_NUM_EU_DOT_COMMA.search(x)) for x in lst)
    return ('.', ',') if eu_hits > en_hits else (',', '.')


def _detect_lexical_tokens(
    stripped_df: pd.DataFrame,
    currency_symbols: Tuple[str, ...] = ("€", "$", "£"),
    iso_currencies: Tuple[str, ...] = ("EUR", "USD", "GBP")
    ) -> pd.DataFrame:
    """
    Detect presence of lexical tokens in DataFrame rows (A2 metrics).

    Args:
        stripped_df: DataFrame with trimmed string values
        currency_symbols: Tuple of currency symbols to detect
        iso_currencies: Tuple of ISO currency codes to detect

    Returns:
        DataFrame with boolean columns for token presence:
        - contains_total_token, contains_avg_token, contains_count_token
        - contains_minmax_token, contains_currency_symbol, contains_iso_currency
    """
    # Compile regex patterns
    re_total = re.compile(r"\b(grand\s+total|subtotal|total|balance|running\s+total|sum)\b", re.I)
    re_avg   = re.compile(r"\b(avg|average|mean)\b", re.I)
    re_cnt   = re.compile(r"\bcount\b", re.I)
    re_minmx = re.compile(r"\b(min|max)\b", re.I)
    iso_pat  = re.compile(r"\b(" + "|".join(map(re.escape, iso_currencies)) + r")\b", re.I)

    # Create currency symbol pattern from input
    curr_pattern = "(" + "|".join(map(re.escape, currency_symbols)) + ")"

    # Apply patterns to detect tokens across all columns
    contains_total_df = stripped_df.apply(lambda col: col.str.contains(re_total, na=False))
    contains_avg_df   = stripped_df.apply(lambda col: col.str.contains(re_avg,   na=False))
    contains_cnt_df   = stripped_df.apply(lambda col: col.str.contains(re_cnt,   na=False))
    contains_minmx_df = stripped_df.apply(lambda col: col.str.contains(re_minmx, na=False))
    contains_curr_df  = stripped_df.apply(lambda col: col.str.contains(curr_pattern, regex=True, na=False))
    contains_iso_df   = stripped_df.apply(lambda col: col.str.contains(iso_pat, na=False))

    # Return row-level boolean indicators
    return pd.DataFrame({
        "contains_total_token":  contains_total_df.any(axis=1),
        "contains_avg_token":    contains_avg_df.any(axis=1),
        "contains_count_token":  contains_cnt_df.any(axis=1),
        "contains_minmax_token": contains_minmx_df.any(axis=1),
        "contains_currency_symbol": contains_curr_df.any(axis=1),
        "contains_iso_currency": contains_iso_df.any(axis=1)
    }, index=stripped_df.index)


def _normalize_numeric_locale_aware(
    stripped_df: pd.DataFrame,
    thousands_used: str,
    decimal_used: str,
    currency_symbols: Tuple[str, ...],
    iso_currencies: Tuple[str, ...]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize and parse numeric values from DataFrame with locale-aware formatting.

    Args:
        stripped_df: DataFrame with trimmed string values
        thousands_used: Thousands separator character
        decimal_used: Decimal separator character
        currency_symbols: Tuple of currency symbols to remove
        iso_currencies: Tuple of ISO currency codes to remove

    Returns:
        Tuple of (numeric_values_df, numeric_mask_df)
        - numeric_values_df: DataFrame with parsed numeric values (NaN for non-numeric)
        - numeric_mask_df: Boolean DataFrame indicating which cells are numeric
    """
    # 1) Strip currency markers & ISO codes; normalize separators; preserve parentheses
    # Also remove thin spaces (U+202F) and NBSP (U+00A0), and regular spaces
    norm = stripped_df.copy()

    # Remove currency symbols one by one
    for sym in currency_symbols:
        norm = norm.replace(re.escape(sym), "", regex=True)

    # Remove ISO currency codes
    iso_pattern = rf"\b({ '|'.join(map(re.escape, iso_currencies)) })\b"
    norm = norm.apply(lambda col: col.str.replace(iso_pattern, "", regex=True))

    # Remove special spaces
    norm = norm.apply(lambda col: col.str.replace("\u202f", "", regex=False))
    norm = norm.apply(lambda col: col.str.replace("\xa0", "", regex=False))
    norm = norm.apply(lambda col: col.str.replace(" ", "", regex=False))
    if thousands_used and thousands_used != decimal_used:
        norm = norm.apply(lambda col: col.str.replace(thousands_used, "", regex=False))
    if decimal_used and decimal_used != ".":
        norm = norm.apply(lambda col: col.str.replace(decimal_used, ".", regex=False))

    # 2) Identify numeric-looking cells (including accounting parentheses), then coerce
    pattern = r"^\(?[+-]?\d+(\.\d+)?\)?$"
    num_like_df = norm.apply(lambda col: col.str.match(pattern, na=False))
    # Apply pd.to_numeric to each column individually
    normalized_for_parsing = norm.where(num_like_df).apply(lambda col: col.str.replace(r"^\((.*)\)$", r"-\1", regex=True))
    numeric_vals_df = normalized_for_parsing.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    numeric_mask_df = numeric_vals_df.notna()

    return numeric_vals_df, numeric_mask_df


def _prepare_dataframe_for_analysis(
    df: pd.DataFrame,
    infer_locale: bool = True,
    thousands: str = ",",
    decimal: str = ".",
    currency_symbols: Tuple[str, ...] = ("€", "$", "£"),
    iso_currencies: Tuple[str, ...] = ("EUR", "USD", "GBP")
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Prepare DataFrame for analysis with locale inference, string normalization, and basic masks.

    Args:
        df: Raw input DataFrame to analyze
        infer_locale: Whether to auto-detect number formatting from data
        thousands: Default thousands separator if not inferring
        decimal: Default decimal separator if not inferring
        currency_symbols: Currency symbols for locale inference sampling
        iso_currencies: ISO currency codes for locale inference sampling

    Returns:
        Tuple of (stripped_df, null_mask_df, date_mask_df, paren_neg_mask_df, thousands_used, decimal_used):
        - stripped_df: DataFrame with all cells converted to trimmed strings
        - null_mask_df: Boolean DataFrame indicating blank/empty cells
        - date_mask_df: Boolean DataFrame indicating date-like cells
        - paren_neg_mask_df: Boolean DataFrame indicating parentheses negatives
        - thousands_used: Actual thousands separator used (inferred or default)
        - decimal_used: Actual decimal separator used (inferred or default)
    """
    # -----------------------------
    # Locale inference (vectorized sampling)
    # -----------------------------
    if infer_locale and not df.empty:
        sample_strings = (
            df.select_dtypes(include=["object", "string"])
              .stack()
              .astype(str)
              .head(20)  # cap deterministically
              .tolist()
        )
        if sample_strings:
            thousands_used, decimal_used = _infer_locale_from_strings(sample_strings)
        else:
            thousands_used, decimal_used = (thousands, decimal)
    else:
        thousands_used, decimal_used = (thousands, decimal)

    # -----------------------------
    # Build normalized string view & null mask once (vectorized)
    # -----------------------------
    sdf = df.astype("string").fillna("")               # all cells as strings ("" for NA)
    stripped = sdf.map(lambda s: s.strip())            # whitespace-trimmed strings
    null_mask_df = (stripped == "")                    # True where blank/empty

    # -----------------------------
    # Date-like detection (vectorized and optimized)
    # -----------------------------
    def _detect_dates_optimized(col):
        """Optimized date detection using vectorized operations."""
        # Skip if column is all empty/null
        if col.isna().all():
            return pd.Series([False] * len(col), index=col.index)

        # Try standard parsing first (most common case)
        parsed_standard = pd.to_datetime(col, errors="coerce")
        mask_standard = parsed_standard.notna()

        # Only try alternative parsing on values that failed standard parsing
        failed_mask = ~mask_standard & col.notna()
        if not failed_mask.any():
            return mask_standard

        # For failed values, try with dayfirst=True (for DD/MM/YYYY formats)
        failed_values = col[failed_mask]
        parsed_dayfirst = pd.to_datetime(failed_values, errors="coerce", dayfirst=True)
        mask_dayfirst = parsed_dayfirst.notna()

        # Combine results
        final_mask = mask_standard.copy()
        final_mask.loc[failed_mask] = mask_dayfirst

        return final_mask

    date_mask_df = stripped.apply(_detect_dates_optimized)

    # -----------------------------
    # Parentheses-negatives (vectorized)
    # -----------------------------
    # Pattern for accounting-style negatives: currency symbol + number in parentheses
    # Examples: ($1,234.56), (€500.00), (£300), (1234.56), (1,000)
    paren_pattern = r"^\([$€£¥]?[\d,]+(?:\.\d+)?\)$"
    paren_neg_df = stripped.apply(lambda col: col.str.contains(paren_pattern, regex=True, na=False))

    return stripped, null_mask_df, date_mask_df, paren_neg_df, thousands_used, decimal_used


def _calculate_density_type_metrics(
    null_mask_df: pd.DataFrame,
    date_mask_df: pd.DataFrame,
    numeric_mask_df: pd.DataFrame,
    df_index: pd.Index,
    n_cols: int
    ) -> pd.DataFrame:
    """
    Calculate M1 density and type mix metrics for each row.

    Args:
        null_mask_df: Boolean DataFrame indicating blank/empty cells
        date_mask_df: Boolean DataFrame indicating date-like cells
        numeric_mask_df: Boolean DataFrame indicating numeric cells
        df_index: Index from original DataFrame for result alignment
        n_cols: Number of columns in original DataFrame

    Returns:
        DataFrame with M1 metrics:
        - non_null_ratio: Fraction of non-blank cells
        - text_count, numeric_count, date_count, null_count: Cell type counts
        - text_ratio, numeric_ratio: Ratios relative to non-null cells
        - type_entropy: Information entropy over cell types
    """
    # Calculate counts for each cell type
    null_count_sr    = null_mask_df.sum(axis=1).astype(int)                # number of blanks in row
    date_count_sr    = date_mask_df.sum(axis=1).astype(int)                # number of date-like cells
    numeric_count_sr = numeric_mask_df.sum(axis=1).astype(int)             # number of numeric cells
    text_mask_df     = (~null_mask_df) & (~date_mask_df) & (~numeric_mask_df)
    text_count_sr    = text_mask_df.sum(axis=1).astype(int)                # number of text-like cells

    # Calculate ratios
    non_null_sr       = (n_cols - null_count_sr)
    non_null_ratio_sr = (non_null_sr / n_cols).fillna(0.0)                 # fraction of non-blanks
    text_ratio_sr     = (text_count_sr / non_null_sr.replace(0, np.nan)).fillna(0.0)
    numeric_ratio_sr  = (numeric_count_sr / non_null_sr.replace(0, np.nan)).fillna(0.0)

    # Type entropy over {blank,text,numeric,date}; low for totals, higher for mixed detail rows
    P_blank = (null_count_sr / n_cols).to_numpy()[:, None]
    P_text  = (text_count_sr / n_cols).to_numpy()[:, None]
    P_num   = (numeric_count_sr / n_cols).to_numpy()[:, None]
    P_date  = (date_count_sr / n_cols).to_numpy()[:, None]
    P = np.hstack([P_blank, P_text, P_num, P_date])
    P = np.clip(P, 1e-12, 1.0)                                             # avoid log(0)
    type_entropy_sr = pd.Series((-(P * np.log(P)).sum(axis=1)), index=df_index)

    return pd.DataFrame({
        "non_null_ratio": non_null_ratio_sr,
        "text_count":     text_count_sr,
        "numeric_count":  numeric_count_sr,
        "date_count":     date_count_sr,
        "null_count":     null_count_sr,
        "text_ratio":     text_ratio_sr,
        "numeric_ratio":  numeric_ratio_sr,
        "type_entropy":   type_entropy_sr,
    }, index=df_index)


def _calculate_numeric_statistics(
    numeric_vals_df: pd.DataFrame,
    paren_neg_present_sr: pd.Series
    ) -> pd.DataFrame:
    """
    Calculate M3 numeric magnitude and sign metrics for each row.

    Args:
        numeric_vals_df: DataFrame with parsed numeric values (NaN for non-numeric)
        paren_neg_present_sr: Boolean Series indicating parentheses negatives per row

    Returns:
        DataFrame with M3 metrics:
        - mean_numeric: Average of numeric cells in row
        - std_numeric: Standard deviation of numeric cells in row
        - p95_numeric: 95th percentile of numeric cells in row
        - max_over_median_ratio: Ratio of max to median (highlights single large numbers)
        - has_parentheses_negative: Boolean indicating presence of parentheses negatives
    """
    # Calculate basic statistics
    mean_numeric_sr = numeric_vals_df.mean(axis=1, skipna=True).fillna(0.0)   # average of numeric cells in row
    std_numeric_sr  = numeric_vals_df.std(axis=1, ddof=0).fillna(0.0)         # stddev of numeric cells
    p95_numeric_sr  = numeric_vals_df.quantile(0.95, axis=1, interpolation="linear").fillna(0.0)  # 95th pct
    median_sr       = numeric_vals_df.median(axis=1).fillna(0.0)              # median of numeric cells
    max_sr          = numeric_vals_df.max(axis=1).fillna(0.0)                 # max numeric in row

    # ratio of "biggest number vs central tendency" → highlights rows with a single large number (totals)
    max_over_median_ratio_sr = (max_sr / (median_sr.abs() + 1e-12)).where(median_sr.ne(0), 0.0)

    return pd.DataFrame({
        "mean_numeric":            mean_numeric_sr,
        "std_numeric":             std_numeric_sr,
        "p95_numeric":             p95_numeric_sr,
        "max_over_median_ratio":   max_over_median_ratio_sr,
        "has_parentheses_negative": paren_neg_present_sr,
    }, index=numeric_vals_df.index)


def _calculate_position_context_metrics(
    null_mask_df: pd.DataFrame,
    non_null_ratio_sr: pd.Series,
    blank_threshold: float,
    df_index: pd.Index,
    n_rows: int
    ) -> pd.DataFrame:
    """
    Calculate M4 position and block context metrics for each row.

    Args:
        null_mask_df: Boolean DataFrame indicating blank/empty cells
        non_null_ratio_sr: Series with fraction of non-blank cells per row
        blank_threshold: Threshold for considering rows as blank-ish
        df_index: Index from original DataFrame for result alignment
        n_rows: Number of rows in original DataFrame

    Returns:
        DataFrame with M4 metrics:
        - row_pos: Normalized position of row (0.0 to 1.0)
        - prev_blank: Boolean indicating if previous row is blank-ish
        - next_blank: Boolean indicating if next row is blank-ish
    """
    # Calculate row position (normalized 0.0 to 1.0) - fix index alignment
    row_pos_sr = (
        pd.Series(np.arange(n_rows), index=df_index, dtype=float) / max(n_rows - 1, 1)
        if n_rows else pd.Series(index=df_index, dtype=float)
    )

    # Detect blank neighbors using shift operations - handle edge rows properly
    prev_blank_sr = (non_null_ratio_sr.shift(1) < blank_threshold).fillna(False)  # no neighbor → False
    next_blank_sr = (non_null_ratio_sr.shift(-1) < blank_threshold).fillna(False) # no neighbor → False

    return pd.DataFrame({
        "row_pos":    row_pos_sr,
        "prev_blank": prev_blank_sr,
        "next_blank": next_blank_sr,
    }, index=df_index)

# -----------------------------
# Vectorized, drop-in replacement of `value_based_memtrics`
# -----------------------------
def value_based_memtrics(
    df: pd.DataFrame,
    infer_locale: bool = True,
    thousands: str = ",",
    decimal: str = ".",
    currency_symbols: Tuple[str, ...] = ("€", "$", "£"),
    iso_currencies: Tuple[str, ...] = ("EUR", "USD", "GBP"),
    blank_threshold: float = 0.15,
    metric_groups: dict = None,
    ) -> pd.DataFrame:
    """
    Vectorized per-row value-based fingerprint metrics (M1–M4) with locale-aware numeric parsing.

    Args:
        df: Input DataFrame to analyze
        infer_locale: Whether to infer locale from data
        thousands: Thousands separator character
        decimal: Decimal separator character
        currency_symbols: Currency symbols to detect and remove
        iso_currencies: ISO currency codes to detect and remove
        blank_threshold: Threshold for considering rows as blank
        metric_groups: Dict specifying which metric groups to calculate
                      {"M1": True, "M2": True, "M3": True, "M4": True}
                      If None, all groups are calculated

    Returns:
        DataFrame with selected metric groups:
          M1: non_null_ratio, text_count, numeric_count, date_count, null_count,
              text_ratio, numeric_ratio, type_entropy
          M2: contains_total_token, contains_avg_token, contains_count_token,
              contains_minmax_token, contains_currency_symbol, contains_iso_currency
          M3: mean_numeric, std_numeric, p95_numeric, max_over_median_ratio,
              has_parentheses_negative
          M4: row_pos, prev_blank, next_blank
        Plus: thousands_used, decimal_used (for provenance/debug).
    """

    # Set default metric groups if none provided
    if metric_groups is None:
        metric_groups = {"M1": True, "M2": True, "M3": True, "M4": True}

    # -----------------------------
    # DataFrame preprocessing pipeline
    # -----------------------------
    stripped, null_mask_df, date_mask_df, paren_neg_df, thousands_used, decimal_used = _prepare_dataframe_for_analysis(
        df, infer_locale, thousands, decimal, currency_symbols, iso_currencies
    )
    paren_neg_present_sr = paren_neg_df.any(axis=1)  # row-level presence

    # -----------------------------
    # Lexical token presence (vectorized)
    # -----------------------------
    # M2: Lexical tokens (row content cues)
    # -----------------------------
    if metric_groups.get("M2", False):
        # Previous (per-row join + regex):
        # joined_lower = " ".join([s for s in str_vals if s != ""]).lower()
        # contains_total_token = bool(re_total.search(joined_lower))
        lexical_tokens_df = _detect_lexical_tokens(stripped, currency_symbols, iso_currencies)
        contains_total_token_sr  = lexical_tokens_df["contains_total_token"]
        contains_avg_token_sr    = lexical_tokens_df["contains_avg_token"]
        contains_count_token_sr  = lexical_tokens_df["contains_count_token"]
        contains_minmax_token_sr = lexical_tokens_df["contains_minmax_token"]
        contains_currency_symbol_sr = lexical_tokens_df["contains_currency_symbol"]
        contains_iso_currency_sr = lexical_tokens_df["contains_iso_currency"]

    # -----------------------------
    # Locale-aware numeric parsing (vectorized, two-stage)
    # -----------------------------
    # Previous (per-cell parse with _parse_numeric_locale):
    # num_val = _parse_numeric_locale(v_str, thousands_used, decimal_used, currency_symbols, iso_currencies)
    numeric_vals_df, numeric_mask_df = _normalize_numeric_locale_aware(
        stripped, thousands_used, decimal_used, currency_symbols, iso_currencies
    )

    # -----------------------------
    # M1: Density & type mix (row "shape")
    # -----------------------------
    if metric_groups.get("M1", False):
        m1_metrics_df = _calculate_density_type_metrics(
            null_mask_df, date_mask_df, numeric_mask_df, df.index, df.shape[1]
        )
        # Extract individual series for use by other metric groups
        non_null_ratio_sr = m1_metrics_df["non_null_ratio"]
        text_count_sr = m1_metrics_df["text_count"]
        numeric_count_sr = m1_metrics_df["numeric_count"]
        date_count_sr = m1_metrics_df["date_count"]
        null_count_sr = m1_metrics_df["null_count"]
        text_ratio_sr = m1_metrics_df["text_ratio"]
        numeric_ratio_sr = m1_metrics_df["numeric_ratio"]
        type_entropy_sr = m1_metrics_df["type_entropy"]

    # -----------------------------
    # A2: Lexical tokens (row content cues)
    # -----------------------------
    # (already computed above as *_sr booleans)

    # -----------------------------
    # M3: Numeric magnitude & sign
    # -----------------------------
    if metric_groups.get("M3", False):
        m3_metrics_df = _calculate_numeric_statistics(numeric_vals_df, paren_neg_present_sr)
        # Extract individual series for use in output assembly
        mean_numeric_sr = m3_metrics_df["mean_numeric"]
        std_numeric_sr = m3_metrics_df["std_numeric"]
        p95_numeric_sr = m3_metrics_df["p95_numeric"]
        max_over_median_ratio_sr = m3_metrics_df["max_over_median_ratio"]

    # -----------------------------
    # M4: Position & block context
    # -----------------------------
    if metric_groups.get("M4", False):
        # M4 depends on non_null_ratio_sr from M1, calculate if needed
        if not metric_groups.get("M1", False):
            null_count_sr = null_mask_df.sum(axis=1).astype(int)
            n_cols = df.shape[1]
            non_null_sr = (n_cols - null_count_sr)
            non_null_ratio_sr = (non_null_sr / n_cols).fillna(0.0)

        m4_metrics_df = _calculate_position_context_metrics(
            null_mask_df, non_null_ratio_sr, blank_threshold, df.index, len(df)
        )
        # Extract individual series for use in output assembly
        row_pos_sr = m4_metrics_df["row_pos"]
        prev_blank_sr = m4_metrics_df["prev_blank"]
        next_blank_sr = m4_metrics_df["next_blank"]

    # -----------------------------
    # Assemble the output DataFrame conditionally based on enabled metrics
    # -----------------------------
    metrics_dict = {
        "row_idx": df.index.to_numpy(),
    }

    # M1 — Density & type mix (row "shape")
    if metric_groups.get("M1", False):
        metrics_dict.update({
            "non_null_ratio": non_null_ratio_sr,
            "text_count":     text_count_sr,
            "numeric_count":  numeric_count_sr,
            "date_count":     date_count_sr,
            "null_count":     null_count_sr,
            "text_ratio":     text_ratio_sr,
            "numeric_ratio":  numeric_ratio_sr,
            "type_entropy":   type_entropy_sr,
        })

    # M2 — Lexical tokens (row content cues)
    if metric_groups.get("M2", False):
        metrics_dict.update({
            "contains_total_token":  contains_total_token_sr,
            "contains_avg_token":    contains_avg_token_sr,
            "contains_count_token":  contains_count_token_sr,
            "contains_minmax_token": contains_minmax_token_sr,
            "contains_currency_symbol": contains_currency_symbol_sr,
            "contains_iso_currency":    contains_iso_currency_sr,
        })

    # M3 — Numeric magnitude & sign
    if metric_groups.get("M3", False):
        metrics_dict.update({
            "mean_numeric":            mean_numeric_sr,
            "std_numeric":             std_numeric_sr,
            "p95_numeric":             p95_numeric_sr,
            "max_over_median_ratio":   max_over_median_ratio_sr,
            "has_parentheses_negative": paren_neg_present_sr,
        })

    # M4 — Position & block context
    if metric_groups.get("M4", False):
        metrics_dict.update({
            "row_pos":    row_pos_sr,
            "prev_blank": prev_blank_sr,
            "next_blank": next_blank_sr,
        })

    # Always include provenance
    metrics_dict.update({
        "thousands_used": thousands_used,
        "decimal_used":   decimal_used,
    })

    metrics = pd.DataFrame(metrics_dict)

    # Ensure expected dtypes (booleans explicitly boolean) - only for included columns
    bool_cols = [
        "contains_total_token", "contains_avg_token", "contains_count_token",
        "contains_minmax_token", "contains_currency_symbol", "contains_iso_currency",
        "has_parentheses_negative", "prev_blank", "next_blank"
    ]
    for c in bool_cols:
        if c in metrics.columns:
            metrics[c] = metrics[c].astype(bool)

    int_cols = ["text_count", "numeric_count", "date_count", "null_count"]
    for c in int_cols:
        if c in metrics.columns:
            metrics[c] = metrics[c].astype(int)

    return metrics
