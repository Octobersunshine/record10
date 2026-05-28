import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple


_PRECISION_UNITS = {
    "s": "s",
    "ms": "ms",
    "millisecond": "ms",
    "us": "us",
    "microsecond": "us",
    "ns": "ns",
    "nanosecond": "ns",
}


def _normalize_index(
    obj: Union[pd.Series, pd.DataFrame],
    precision: str,
) -> Union[pd.Series, pd.DataFrame]:
    unit = _PRECISION_UNITS.get(precision)
    if unit is None:
        raise ValueError(
            f"precision must be one of {list(_PRECISION_UNITS.keys())}, got '{precision}'"
        )
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("precision normalization requires a DatetimeIndex")
    rounded = obj.index.round(unit)
    result = obj.copy()
    result.index = rounded
    return result


def _apply_fill_value(
    df: pd.DataFrame,
    fill_value: Optional[Union[float, dict]],
) -> pd.DataFrame:
    if fill_value is None:
        return df
    if isinstance(fill_value, dict):
        for col, val in fill_value.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
    else:
        df = df.fillna(fill_value)
    return df


def _tolerance_merge_series(
    left: pd.Series,
    right: pd.Series,
    how: str,
    tolerance: pd.Timedelta,
    suffixes: tuple,
) -> pd.DataFrame:
    left_sorted = left.sort_index()
    right_sorted = right.sort_index()

    left_name = left.name or "value"
    right_name = right.name or "value"

    same_name = left_name == right_name
    if same_name:
        left_name = left_name + suffixes[0]
        right_name = right_name + suffixes[1]

    left_df = left_sorted.to_frame(name=left_name)
    right_df = right_sorted.to_frame(name=right_name)

    if how in ("left", "inner"):
        merged = pd.merge_asof(
            left_df, right_df,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
        )
        if how == "inner":
            merged = merged.dropna(subset=[right_name])

    elif how == "right":
        merged_rev = pd.merge_asof(
            right_df, left_df,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
        )
        merged = merged_rev[[left_name, right_name]]

    elif how == "outer":
        merged_left = pd.merge_asof(
            left_df, right_df,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
        )
        merged_right = pd.merge_asof(
            right_df, left_df,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
        )

        right_only_mask = merged_right[left_name].isna()
        right_only = merged_right.loc[right_only_mask, [right_name]]

        merged = pd.concat([merged_left, right_only])
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        merged = merged[[left_name, right_name]]

    else:
        raise ValueError(f"how must be one of 'inner','outer','left','right', got '{how}'")

    return merged


def _tolerance_merge_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str,
    tolerance: pd.Timedelta,
    suffixes: tuple,
) -> pd.DataFrame:
    left_sorted = left.sort_index()
    right_sorted = right.sort_index()

    if how in ("left", "inner"):
        merged = pd.merge_asof(
            left_sorted, right_sorted,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
            suffixes=suffixes,
        )
        if how == "inner":
            right_cols = [c for c in merged.columns if any(
                c.endswith(s) for s in suffixes
            ) or c in right_sorted.columns]
            right_cols_in_merged = [c for c in right_cols if c in merged.columns]
            if right_cols_in_merged:
                merged = merged.dropna(subset=right_cols_in_merged)

    elif how == "right":
        merged = pd.merge_asof(
            right_sorted, left_sorted,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
            suffixes=suffixes,
        )
        left_cols = [c for c in merged.columns if c in left_sorted.columns or any(
            c.endswith(suffixes[0]) and c not in right_sorted.columns
            for c in merged.columns
        )]
        right_cols = [c for c in merged.columns if c in right_sorted.columns or any(
            c.endswith(suffixes[1]) and c not in left_sorted.columns
            for c in merged.columns
        )]
        ordered = [c for c in left_cols + right_cols if c in merged.columns]
        merged = merged[ordered]

    elif how == "outer":
        merged_left = pd.merge_asof(
            left_sorted, right_sorted,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
            suffixes=suffixes,
        )
        merged_right = pd.merge_asof(
            right_sorted, left_sorted,
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=tolerance,
            suffixes=suffixes,
        )

        left_matched_idx = merged_left.index
        right_only_mask = ~merged_right.index.isin(left_matched_idx)
        right_only = merged_right[right_only_mask]

        merged = pd.concat([merged_left, right_only])
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]

    else:
        raise ValueError(f"how must be one of 'inner','outer','left','right', got '{how}'")

    return merged


def ts_join(
    left: pd.Series,
    right: pd.Series,
    how: str = "inner",
    fill_value: Optional[Union[float, dict]] = None,
    suffixes: tuple = ("_left", "_right"),
    precision: Optional[str] = None,
    tolerance: Optional[Union[pd.Timedelta, str]] = None,
) -> pd.DataFrame:
    if not isinstance(left, pd.Series):
        raise TypeError(f"left must be pd.Series, got {type(left)}")
    if not isinstance(right, pd.Series):
        raise TypeError(f"right must be pd.Series, got {type(right)}")
    if how not in ("inner", "outer", "left", "right"):
        raise ValueError(f"how must be one of 'inner','outer','left','right', got '{how}'")

    if precision is not None:
        left = _normalize_index(left, precision)
        right = _normalize_index(right, precision)

    if isinstance(tolerance, str):
        tolerance = pd.Timedelta(tolerance)

    if tolerance is not None:
        merged = _tolerance_merge_series(left, right, how, tolerance, suffixes)
    else:
        left_df = left.to_frame(name=left.name or "value")
        right_df = right.to_frame(name=right.name or "value")

        if left_df.columns[0] == right_df.columns[0]:
            right_df.columns = [right_df.columns[0] + suffixes[1]]
            left_df.columns = [left_df.columns[0] + suffixes[0]]
        else:
            right_df.columns = [right_df.columns[0]]
            left_df.columns = [left_df.columns[0]]

        merged = pd.merge(
            left_df, right_df,
            left_index=True,
            right_index=True,
            how=how,
        )

    merged = _apply_fill_value(merged, fill_value)
    return merged


def ts_join_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    fill_value: Optional[Union[float, dict]] = None,
    suffixes: tuple = ("_left", "_right"),
    precision: Optional[str] = None,
    tolerance: Optional[Union[pd.Timedelta, str]] = None,
) -> pd.DataFrame:
    if not isinstance(left, pd.DataFrame):
        raise TypeError(f"left must be pd.DataFrame, got {type(left)}")
    if not isinstance(right, pd.DataFrame):
        raise TypeError(f"right must be pd.DataFrame, got {type(right)}")
    if how not in ("inner", "outer", "left", "right"):
        raise ValueError(f"how must be one of 'inner','outer','left','right', got '{how}'")

    if precision is not None:
        left = _normalize_index(left, precision)
        right = _normalize_index(right, precision)

    if isinstance(tolerance, str):
        tolerance = pd.Timedelta(tolerance)

    if tolerance is not None:
        merged = _tolerance_merge_dataframes(left, right, how, tolerance, suffixes)
    else:
        merged = pd.merge(
            left, right,
            left_index=True,
            right_index=True,
            how=how,
            suffixes=suffixes,
        )

    merged = _apply_fill_value(merged, fill_value)
    return merged


def ts_join_asof(
    left: Union[pd.Series, pd.DataFrame],
    right: Union[pd.Series, pd.DataFrame],
    direction: str = "nearest",
    tolerance: Optional[Union[pd.Timedelta, str]] = None,
    fill_value: Optional[Union[float, dict]] = None,
    suffixes: tuple = ("_left", "_right"),
    precision: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(left, (pd.Series, pd.DataFrame)):
        raise TypeError(f"left must be pd.Series or pd.DataFrame, got {type(left)}")
    if not isinstance(right, (pd.Series, pd.DataFrame)):
        raise TypeError(f"right must be pd.Series or pd.DataFrame, got {type(right)}")

    if precision is not None:
        left = _normalize_index(left, precision)
        right = _normalize_index(right, precision)

    if isinstance(tolerance, str):
        tolerance = pd.Timedelta(tolerance)

    if isinstance(left, pd.Series):
        left_df = left.sort_index().to_frame(name=left.name or "value_left")
    else:
        left_df = left.sort_index()

    if isinstance(right, pd.Series):
        right_df = right.sort_index().to_frame(name=right.name or "value_right")
    else:
        right_df = right.sort_index()

    overlapping = left_df.columns.intersection(right_df.columns)
    if len(overlapping) > 0:
        rename_map = {c: c + suffixes[1] for c in right_df.columns}
        right_df = right_df.rename(columns=rename_map)
        rename_map_left = {c: c + suffixes[0] for c in overlapping}
        left_df = left_df.rename(columns=rename_map_left)

    merged = pd.merge_asof(
        left_df, right_df,
        left_index=True,
        right_index=True,
        direction=direction,
        tolerance=tolerance,
    )

    merged = _apply_fill_value(merged, fill_value)
    return merged


def ts_resample(
    ts: Union[pd.Series, pd.DataFrame],
    freq: str,
    agg_method: str = "mean",
    fill_method: Optional[str] = None,
    limit: Optional[int] = None,
) -> Union[pd.Series, pd.DataFrame]:
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        raise TypeError(f"ts must be pd.Series or pd.DataFrame, got {type(ts)}")
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise TypeError("resample requires a DatetimeIndex")

    resampler = ts.resample(freq)

    if callable(agg_method):
        resampled = resampler.apply(agg_method)
    elif isinstance(agg_method, dict):
        resampled = resampler.agg(agg_method)
    else:
        resampled = getattr(resampler, agg_method)()

    if fill_method is not None:
        resampled = resampled.interpolate(method=fill_method, limit=limit)

    return resampled


def ts_join_multi(
    series_list: List[pd.Series],
    names: Optional[List[str]] = None,
    how: str = "outer",
    fill_value: Optional[Union[float, dict]] = None,
    precision: Optional[str] = None,
    tolerance: Optional[Union[pd.Timedelta, str]] = None,
    resample_freq: Optional[str] = None,
    resample_agg: str = "mean",
) -> pd.DataFrame:
    if not isinstance(series_list, (list, tuple)):
        raise TypeError(f"series_list must be list or tuple, got {type(series_list)}")
    if len(series_list) < 2:
        raise ValueError("series_list must contain at least 2 series")
    if names is not None and len(names) != len(series_list):
        raise ValueError("names length must match series_list length")

    processed = []
    for i, s in enumerate(series_list):
        if not isinstance(s, pd.Series):
            raise TypeError(f"series_list[{i}] must be pd.Series, got {type(s)}")
        s_copy = s.copy()
        if names is not None:
            s_copy.name = names[i]
        elif s_copy.name is None:
            s_copy.name = f"series_{i}"
        processed.append(s_copy)

    if resample_freq is not None:
        processed = [
            ts_resample(s, freq=resample_freq, agg_method=resample_agg)
            for s in processed
        ]

    merged = processed[0].to_frame(name=processed[0].name)

    for i in range(1, len(processed)):
        right = processed[i]
        merged = ts_join_dataframes(
            merged,
            right.to_frame(name=right.name),
            how=how,
            fill_value=None,
            precision=precision,
            tolerance=tolerance,
        )

    merged = _apply_fill_value(merged, fill_value)
    return merged


def coverage_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be pd.DataFrame, got {type(df)}")

    total_rows = len(df)
    non_null = df.notna().sum()
    null = df.isna().sum()
    coverage_pct = (non_null / total_rows * 100).round(2)

    stats = pd.DataFrame({
        "total_rows": total_rows,
        "non_null_count": non_null,
        "null_count": null,
        "coverage_pct": coverage_pct,
    })
    stats.index.name = "column"
    return stats.sort_values("coverage_pct", ascending=False)


if __name__ == "__main__":
    dates_a = pd.DatetimeIndex([
        "2024-01-01", "2024-01-03", "2024-01-05",
        "2024-01-07", "2024-01-09",
    ])
    dates_b = pd.DatetimeIndex([
        "2024-01-02", "2024-01-03", "2024-01-05",
        "2024-01-08", "2024-01-10",
    ])

    ts_a = pd.Series([10, 20, 30, 40, 50], index=dates_a, name="price_a")
    ts_b = pd.Series([100, 200, 300, 400, 500], index=dates_b, name="price_b")

    print("=" * 60)
    print("Time Series A:")
    print(ts_a)
    print("\nTime Series B:")
    print(ts_b)

    print("\n" + "=" * 60)
    print("INNER JOIN (only matching timestamps)")
    print(ts_join(ts_a, ts_b, how="inner"))

    print("\n" + "=" * 60)
    print("OUTER JOIN (all timestamps, NaN for missing)")
    print(ts_join(ts_a, ts_b, how="outer"))

    print("\n" + "=" * 60)
    print("LEFT JOIN (keep all from A)")
    print(ts_join(ts_a, ts_b, how="left"))

    print("\n" + "=" * 60)
    print("RIGHT JOIN (keep all from B)")
    print(ts_join(ts_a, ts_b, how="right"))

    print("\n" + "=" * 60)
    print("OUTER JOIN with fill_value=0")
    print(ts_join(ts_a, ts_b, how="outer", fill_value=0))

    print("\n" + "=" * 60)
    print("OUTER JOIN with dict fill_value per column")
    print(ts_join(ts_a, ts_b, how="outer", fill_value={"price_a": -1, "price_b": -999}))

    print("\n" + "=" * 60)
    print("PRECISION MISMATCH FIX")
    print("-" * 40)

    ts_ms = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex([
            "2024-01-01 00:00:00.123",
            "2024-01-01 00:00:01.456",
            "2024-01-01 00:00:02.789",
        ]),
        name="ts_ms",
    )
    ts_us = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.DatetimeIndex([
            "2024-01-01 00:00:00.123000",
            "2024-01-01 00:00:01.456789",
            "2024-01-01 00:00:02.789123",
        ]),
        name="ts_us",
    )

    print("Series with millisecond precision:")
    print(ts_ms)
    print("\nSeries with microsecond precision (slightly different sub-second values):")
    print(ts_us)

    print("\nWithout precision normalization — INNER JOIN yields 1 match only:")
    print(ts_join(ts_ms, ts_us, how="inner"))

    print("\nWith precision='ms' — INNER JOIN rounds both to ms (1.456789 rounds to 1.457, so 2/3 match):")
    print(ts_join(ts_ms, ts_us, how="inner", precision="ms"))

    print("\nWith tolerance='1ms' — INNER JOIN catches all 3 (±1ms fuzzy match):")
    print(ts_join(ts_ms, ts_us, how="inner", tolerance="1ms"))

    print("\nCombine precision='ms' + tolerance='500us' for best of both worlds:")
    print(ts_join(ts_ms, ts_us, how="inner", precision="ms", tolerance="500us"))

    print("\n" + "=" * 60)
    print("TOLERANCE-BASED ALIGNMENT (±δ)")
    print("-" * 40)

    ts_x = pd.Series(
        [100, 200, 300, 400],
        index=pd.DatetimeIndex([
            "2024-03-01 10:00:00.000",
            "2024-03-01 10:00:01.000",
            "2024-03-01 10:00:02.000",
            "2024-03-01 10:00:05.000",
        ]),
        name="ts_x",
    )
    ts_y = pd.Series(
        [1000, 2000, 3000, 4000],
        index=pd.DatetimeIndex([
            "2024-03-01 10:00:00.200",
            "2024-03-01 10:00:00.900",
            "2024-03-01 10:00:02.100",
            "2024-03-01 10:00:06.000",
        ]),
        name="ts_y",
    )

    print("Series X (timestamps at .000, .000, .000, .000):")
    print(ts_x)
    print("\nSeries Y (timestamps slightly offset):")
    print(ts_y)

    print("\nExact INNER JOIN — no matches (timestamps differ):")
    print(ts_join(ts_x, ts_y, how="inner"))

    print("\nWith tolerance='500ms' — INNER JOIN matches within ±500ms:")
    print(ts_join(ts_x, ts_y, how="inner", tolerance="500ms"))

    print("\nWith tolerance='500ms' — LEFT JOIN (keep all X, match Y within ±500ms):")
    print(ts_join(ts_x, ts_y, how="left", tolerance="500ms"))

    print("\nWith tolerance='500ms' — OUTER JOIN:")
    print(ts_join(ts_x, ts_y, how="outer", tolerance="500ms", fill_value=0))

    print("\nWith tolerance='2s' — LEFT JOIN (wider tolerance, more matches):")
    print(ts_join(ts_x, ts_y, how="left", tolerance="2s"))

    print("\n" + "=" * 60)
    print("ASOF JOIN — nearest neighbor match")
    print("-" * 40)

    ts_c = pd.Series([15, 25, 35, 45, 55], index=dates_a, name="price_c")
    print("ASOF JOIN (nearest, no tolerance):")
    print(ts_join_asof(ts_b, ts_c, direction="nearest"))

    print("\nASOF JOIN with tolerance=1 day:")
    print(ts_join_asof(ts_b, ts_c, direction="nearest", tolerance=pd.Timedelta("1 day"), fill_value=0))

    print("\nASOF JOIN — DataFrame support:")
    df_left = pd.DataFrame(
        {"bid": [1.1, 1.2, 1.3], "ask": [1.15, 1.25, 1.35]},
        index=pd.DatetimeIndex([
            "2024-06-01 09:30:00",
            "2024-06-01 09:30:05",
            "2024-06-01 09:30:10",
        ]),
    )
    df_right = pd.DataFrame(
        {"bid": [1.11, 1.22, 1.33], "ask": [1.16, 1.26, 1.36]},
        index=pd.DatetimeIndex([
            "2024-06-01 09:30:01",
            "2024-06-01 09:30:06",
            "2024-06-01 09:30:11",
        ]),
    )
    print("Left DataFrame:")
    print(df_left)
    print("\nRight DataFrame (1 second offset):")
    print(df_right)
    print("\nASOF JOIN with precision='s' (round to second before asof):")
    print(ts_join_asof(df_left, df_right, direction="nearest", precision="s"))

    print("\n" + "=" * 60)
    print("DataFrame LEFT JOIN demo (exact match)")
    df_a = pd.DataFrame(
        {"temp": [22, 24, 20, 25, 23], "humidity": [60, 55, 70, 50, 65]},
        index=dates_a,
    )
    df_b = pd.DataFrame(
        {"wind": [5, 10, 3, 8, 12], "rain": [0, 1, 0, 2, 0]},
        index=dates_b,
    )
    print(ts_join_dataframes(df_a, df_b, how="left", fill_value=0))

    print("\n" + "=" * 60)
    print("MULTI-SERIES ALIGNMENT (N > 2)")
    print("-" * 40)

    ts_1 = pd.Series(
        [10, 20, 30, 40, 50],
        index=pd.DatetimeIndex([
            "2024-04-01 09:00",
            "2024-04-01 09:05",
            "2024-04-01 09:10",
            "2024-04-01 09:15",
            "2024-04-01 09:20",
        ]),
        name="sensor_a",
    )
    ts_2 = pd.Series(
        [100, 200, 300, 400],
        index=pd.DatetimeIndex([
            "2024-04-01 09:00",
            "2024-04-01 09:02",
            "2024-04-01 09:10",
            "2024-04-01 09:12",
        ]),
        name="sensor_b",
    )
    ts_3 = pd.Series(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        index=pd.DatetimeIndex([
            "2024-04-01 09:00",
            "2024-04-01 09:05",
            "2024-04-01 09:07",
            "2024-04-01 09:10",
            "2024-04-01 09:15",
            "2024-04-01 09:18",
        ]),
        name="sensor_c",
    )
    ts_4 = pd.Series(
        [99, 88, 77],
        index=pd.DatetimeIndex([
            "2024-04-01 09:00",
            "2024-04-01 09:10",
            "2024-04-01 09:20",
        ]),
        name="sensor_d",
    )

    print("Sensor A (5 samples):")
    print(ts_1)
    print("\nSensor B (4 samples):")
    print(ts_2)
    print("\nSensor C (6 samples):")
    print(ts_3)
    print("\nSensor D (3 samples):")
    print(ts_4)

    print("\nOuter join of all 4 series (union of all timestamps:")
    multi_outer = ts_join_multi(
        [ts_1, ts_2, ts_3, ts_4],
        how="outer",
    )
    print(multi_outer)
    print("\nCoverage statistics:")
    print(coverage_stats(multi_outer))

    print("\nInner join of all 4 series (intersection of timestamps only:")
    multi_inner = ts_join_multi(
        [ts_1, ts_2, ts_3, ts_4],
        how="inner",
    )
    print(multi_inner)

    print("\n" + "=" * 60)
    print("MIXED FREQUENCY RESAMPLING + ALIGNMENT")
    print("-" * 40)

    ts_hourly = pd.Series(
    [100, 101, 102, 103],
    index=pd.date_range("2024-05-01 00:00", periods=4, freq="h"),
    name="hourly",
)
    ts_30min = pd.Series(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    index=pd.date_range("2024-05-01 00:00", periods=9, freq="30min"),
    name="30min",
)
    ts_15min = pd.Series(
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
    index=pd.date_range("2024-05-01 00:00", periods=17, freq="15min"),
    name="15min",
)

    print("Hourly series:")
    print(ts_hourly)
    print("\n30min series:")
    print(ts_30min)
    print("\n15min series:")
    print(ts_15min)

    print("\nResample all to 15min grid (mean agg):")
    resampled = ts_join_multi(
        [ts_hourly, ts_30min, ts_15min],
        resample_freq="15min",
        resample_agg="mean",
        fill_value=0,
    )
    print(resampled)
    print("\nCoverage after resampling:")
    print(coverage_stats(resampled))

    print("\nResample all to 1h grid (sum agg for 30min and 15min):")
    resampled_h = ts_join_multi(
        [ts_hourly, ts_30min, ts_15min],
        resample_freq="1h",
        resample_agg="sum",
    )
    print(resampled_h)
    print("\nCoverage after 1h resampling:")
    print(coverage_stats(resampled_h))

    print("\n" + "=" * 60)
    print("TOLERANCE + MULTI-SERIES ALIGNMENT")
    print("-" * 40)

    ts_t1 = pd.Series(
    [1.0, 2.0, 3.0],
    index=pd.DatetimeIndex([
        "2024-06-01 12:00:00.000",
        "2024-06-01 12:00:01.000",
        "2024-06-01 12:00:02.000",
    ]),
    name="device_1ms",
)
    ts_t2 = pd.Series(
    [10.0, 20.0, 30.0],
    index=pd.DatetimeIndex([
        "2024-06-01 12:00:00.100",
        "2024-06-01 12:00:01.150",
        "2024-06-01 12:00:02.050",
    ]),
    name="device_2",
)
    ts_t3 = pd.Series(
    [100.0, 200.0, 300.0],
    index=pd.DatetimeIndex([
        "2024-06-01 12:00:00.200",
        "2024-06-01 12:00:00.900",
        "2024-06-01 12:00:02.200",
    ]),
    name="device_3",
)

    print("Without tolerance — inner join (no exact matches:")
    no_tol = ts_join_multi([ts_t1, ts_t2, ts_t3], how="inner")
    print(no_tol)
    print(f"  Rows:", len(no_tol))

    print("\nWith tolerance=500ms tolerance — inner join (fuzzy matches:")
    with_tol = ts_join_multi([ts_t1, ts_t2, ts_t3], how="inner", tolerance="500ms")
    print(with_tol)
    print(f"  Rows:", len(with_tol))
    print("\nCoverage:")
    print(coverage_stats(with_tol))
