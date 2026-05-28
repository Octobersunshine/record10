import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def numeric_binning(df, value_col, bins, bin_col='bin', labels=None, right=False, include_upper=True):
    """
    基于数值范围分箱，统一使用左闭右开规则 [low, high)
    
    参数:
        df: DataFrame
        value_col: 要分箱的数值列名
        bins: 分箱边界列表，如 [0, 18, 30, 45, 60, 100]
        bin_col: 分箱结果列名
        labels: 分箱标签列表，如 ['0-18', '18-30', '30-45', '45-60', '60+']
        right: 布尔值，控制区间方向
            False (默认): 左闭右开 [low, high)
            True: 左开右闭 (low, high]
        include_upper: 布尔值，是否将最大值包含到最后一个区间中
            当 right=False (左闭右开) 时，默认 True，确保最大值被包含
    
    返回:
        添加了分箱列的DataFrame
    """
    df = df.copy()
    bins = np.asarray(bins, dtype=np.float64)
    
    if not right and include_upper:
        max_val = df[value_col].max()
        if max_val >= bins[-1]:
            bins = bins.copy()
            bins[-1] = max_val + 1e-10
    
    df[bin_col] = pd.cut(
        df[value_col], 
        bins=bins, 
        labels=labels, 
        right=right,
        include_lowest=True
    )
    return df


def timestamp_binning(df, timestamp_col, freq='h', bin_col='time_bin'):
    """
    基于时间戳分箱
    
    参数:
        df: DataFrame
        timestamp_col: 时间戳列名
        freq: 分箱频率
            'h' - 按小时
            'D' - 按天
            'W' - 按周
            'ME' - 按月
            'QE' - 按季度
            'YE' - 按年
            也可以使用自定义频率，如 '2h' 表示每2小时
        bin_col: 分箱结果列名
    
    返回:
        添加了时间分箱列的DataFrame
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df[bin_col] = df[timestamp_col].dt.floor(freq)
    return df


def aggregate_bins(df, group_col, value_cols, agg_funcs=None):
    """
    对每个箱内的值进行聚合
    
    参数:
        df: DataFrame
        group_col: 分组列名（分箱列）
        value_cols: 需要聚合的数值列名或列表
        agg_funcs: 聚合函数列表，默认为 ['count', 'sum', 'mean', 'max', 'min']
    
    返回:
        聚合结果DataFrame
    """
    if agg_funcs is None:
        agg_funcs = ['count', 'sum', 'mean', 'max', 'min']
    
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    agg_dict = {col: agg_funcs for col in value_cols}
    
    result = df.groupby(group_col, observed=False).agg(agg_dict).reset_index()
    
    result.columns = [f"{col}_{func}" if col != group_col else col 
                      for col, func in result.columns]
    
    return result


def custom_binning(df, bin_func, bin_col='custom_bin', apply_on=None):
    """
    自定义分箱函数，支持 lambda 表达式
    
    参数:
        df: DataFrame
        bin_func: 分箱函数，返回分箱标签
            接收整行(Series)作为参数，可访问多列:
            lambda row: '高' if row['income'] > 30000 else '低'
        bin_col: 分箱结果列名
        apply_on: 可选，指定列名时 bin_func 仅接收该列的值而非整行
            如 apply_on='income': lambda x: '高' if x > 30000 else '低'
    
    返回:
        添加了分箱列的DataFrame
    
    示例:
        # 基于 score 和 income 多列判断等级
        custom_binning(df, lambda r: 'A' if r['score'] >= 90 and r['income'] > 30000 else 'B')
        
        # 仅基于 income 单列分箱
        custom_binning(df, lambda x: '高' if x > 30000 else '中' if x > 10000 else '低', apply_on='income')
    """
    df = df.copy()
    if apply_on is not None:
        df[bin_col] = df[apply_on].apply(bin_func)
    else:
        df[bin_col] = df.apply(bin_func, axis=1)
    return df


def multi_level_aggregate(df, group_cols, value_cols, agg_funcs=None):
    """
    多级分箱聚合，先按A分箱，再按B分箱
    
    参数:
        df: DataFrame
        group_cols: 分组列名列表，按顺序表示多级分箱层级，如 ['age_bin', 'income_bin']
        value_cols: 需要聚合的数值列名或列表
        agg_funcs: 聚合函数列表，默认为 ['count', 'sum', 'mean', 'max', 'min']
    
    返回:
        多级聚合结果DataFrame
    
    示例:
        # 先按年龄段分箱，再按收入等级分箱
        multi_level_aggregate(df, ['age_bin', 'income_bin'], ['score'])
    """
    if agg_funcs is None:
        agg_funcs = ['count', 'sum', 'mean', 'max', 'min']
    
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    agg_dict = {col: agg_funcs for col in value_cols}
    
    result = df.groupby(group_cols, observed=False).agg(agg_dict).reset_index()
    
    result.columns = [f"{col}_{func}" if col not in group_cols else col 
                      for col, func in result.columns]
    
    return result


def pivot_aggregate(df, index_col, columns_col, value_col, agg_func='mean', fill_value=None, margins=False, margins_name='All'):
    """
    透视表形式输出，行列交叉统计
    
    参数:
        df: DataFrame
        index_col: 行分箱列名
        columns_col: 列分箱列名
        value_col: 需要聚合的数值列名
        agg_func: 聚合函数，支持字符串或列表，如 'mean', 'sum', ['mean', 'sum']
        fill_value: 填充缺失值的值
        margins: 是否添加行/列合计
        margins_name: 合计行/列的名称
    
    返回:
        透视表DataFrame
    
    示例:
        # 行=年龄段，列=收入等级，值=平均分数
        pivot_aggregate(df, 'age_bin', 'income_bin', 'score', agg_func='mean')
    """
    pivot = pd.pivot_table(
        df,
        index=index_col,
        columns=columns_col,
        values=value_col,
        aggfunc=agg_func,
        fill_value=fill_value,
        margins=margins,
        margins_name=margins_name
    )
    
    return pivot


def generate_sample_numeric_data(n=1000):
    """生成示例数值数据"""
    np.random.seed(42)
    data = {
        'age': np.random.randint(1, 100, n),
        'income': np.random.randint(2000, 50000, n),
        'score': np.random.uniform(0, 100, n).round(2)
    }
    return pd.DataFrame(data)


def generate_sample_timestamp_data(n=1000):
    """生成示例时间戳数据"""
    np.random.seed(42)
    base_time = datetime(2024, 1, 1)
    
    timestamps = [base_time + timedelta(minutes=np.random.randint(0, 30 * 24 * 60)) 
                  for _ in range(n)]
    
    data = {
        'timestamp': timestamps,
        'sales': np.random.uniform(10, 500, n).round(2),
        'quantity': np.random.randint(1, 20, n),
        'customer_id': np.random.randint(1, 100, n)
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 60)
    print("示例1: 边界值归属验证（左闭右开 vs 左开右闭）")
    print("=" * 60)
    
    test_data = pd.DataFrame({
        'value': [0, 10, 18, 18.0, 30, 30.0, 45, 60, 100, 100.0]
    })
    bins = [0, 18, 30, 45, 60, 100]
    
    print("\n测试边界值数据:")
    print(test_data.to_string(index=False))
    
    print("\n左闭右开 [low, high) - 默认行为:")
    df_left_closed = numeric_binning(test_data, 'value', bins=bins, right=False, include_upper=False)
    print(df_left_closed.to_string(index=False))
    
    print("\n左开右闭 (low, high]:")
    df_right_closed = numeric_binning(test_data, 'value', bins=bins, right=True)
    print(df_right_closed.to_string(index=False))
    
    print("\n左闭右开 [low, high) - 自动扩展上边界包含最大值:")
    df_include_upper = numeric_binning(test_data, 'value', bins=bins, right=False, include_upper=True)
    print(df_include_upper.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例2: 数值范围分箱（年龄段，左闭右开 [low, high)）")
    print("=" * 60)
    
    df_numeric = generate_sample_numeric_data(500)
    print("\n原始数据（前5行）:")
    print(df_numeric.head())
    
    age_bins = [0, 18, 30, 45, 60, 100]
    age_labels = ['0-18', '18-30', '30-45', '46-60', '60+']
    
    df_binned = numeric_binning(df_numeric, 'age', bins=age_bins, labels=age_labels)
    print("\n分箱后的数据（前5行）:")
    print(df_binned.head())
    
    agg_result = aggregate_bins(df_binned, 'bin', ['income', 'score'])
    print("\n各年龄段聚合结果:")
    print(agg_result.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例3: 时间戳分箱（按小时、按天）")
    print("=" * 60)
    
    df_time = generate_sample_timestamp_data(500)
    print("\n原始数据（前5行）:")
    print(df_time.head())
    
    df_hourly = timestamp_binning(df_time, 'timestamp', freq='h')
    print("\n按小时分箱后的数据（前5行）:")
    print(df_hourly[['timestamp', 'time_bin', 'sales', 'quantity']].head())
    
    hourly_agg = aggregate_bins(df_hourly, 'time_bin', ['sales', 'quantity'])
    print("\n按小时聚合结果（前10行）:")
    print(hourly_agg.head(10).to_string(index=False))
    
    df_daily = timestamp_binning(df_time, 'timestamp', freq='D')
    daily_agg = aggregate_bins(df_daily, 'time_bin', ['sales', 'quantity'])
    print("\n按天聚合结果:")
    print(daily_agg.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例4: 自定义频率分箱（每6小时）")
    print("=" * 60)
    
    df_6h = timestamp_binning(df_time, 'timestamp', freq='6h', bin_col='bin_6h')
    agg_6h = aggregate_bins(df_6h, 'bin_6h', ['sales', 'quantity'])
    print("\n每6小时聚合结果（前10行）:")
    print(agg_6h.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例5: 自定义聚合函数")
    print("=" * 60)
    
    custom_agg = aggregate_bins(
        df_binned, 
        'bin', 
        ['income', 'score'],
        agg_funcs=['count', 'mean', 'median', 'std', 'sum']
    )
    print("\n自定义聚合结果（包含中位数和标准差）:")
    print(custom_agg.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例6: 自定义分箱函数（lambda 表达式）")
    print("=" * 60)
    
    df_custom = generate_sample_numeric_data(200)
    
    df_custom = custom_binning(
        df_custom,
        lambda x: '高' if x > 30000 else '中' if x > 15000 else '低',
        bin_col='income_level',
        apply_on='income'
    )
    print("\n基于 income 单列自定义分箱（前10行）:")
    print(df_custom[['age', 'income', 'score', 'income_level']].head(10))
    
    df_custom = custom_binning(
        df_custom,
        lambda r: '优秀' if r['score'] >= 80 and r['income'] > 25000 
                  else '良好' if r['score'] >= 60 
                  else '待提升',
        bin_col='grade'
    )
    print("\n基于 score 和 income 多列综合分箱（前10行）:")
    print(df_custom[['age', 'income', 'score', 'income_level', 'grade']].head(10))
    
    custom_agg_result = aggregate_bins(df_custom, 'grade', ['score'])
    print("\n按 grade 分箱聚合结果:")
    print(custom_agg_result.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例7: 多级分箱聚合（先按年龄段，再按收入等级）")
    print("=" * 60)
    
    df_multi = generate_sample_numeric_data(500)
    df_multi = numeric_binning(df_multi, 'age', bins=[0, 18, 30, 45, 60, 100], 
                                labels=['0-18', '18-30', '30-45', '45-60', '60+'], bin_col='age_bin')
    df_multi = custom_binning(
        df_multi,
        lambda x: '高收入' if x > 30000 else '中收入' if x > 15000 else '低收入',
        bin_col='income_bin',
        apply_on='income'
    )
    print("\n多级分箱后数据（前5行）:")
    print(df_multi[['age', 'income', 'score', 'age_bin', 'income_bin']].head())
    
    multi_result = multi_level_aggregate(df_multi, ['age_bin', 'income_bin'], ['score'],
                                          agg_funcs=['count', 'mean', 'max', 'min'])
    print("\n多级聚合结果（年龄段 x 收入等级）:")
    print(multi_result.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("示例8: 透视表形式输出（行列交叉统计）")
    print("=" * 60)
    
    pivot_mean = pivot_aggregate(df_multi, 'age_bin', 'income_bin', 'score', agg_func='mean')
    print("\n平均分数 透视表（行=年龄段，列=收入等级）:")
    print(pivot_mean.to_string())
    
    pivot_count = pivot_aggregate(df_multi, 'age_bin', 'income_bin', 'score', agg_func='count', fill_value=0)
    print("\n计数 透视表（行=年龄段，列=收入等级）:")
    print(pivot_count.to_string())
    
    pivot_sum = pivot_aggregate(df_multi, 'age_bin', 'income_bin', 'score', 
                                 agg_func=['mean', 'sum'], margins=True, margins_name='合计')
    print("\n均值+求和 透视表（含行列合计）:")
    print(pivot_sum.to_string())
