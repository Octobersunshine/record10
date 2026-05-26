import numpy as np


def dec_inc_to_vgp(dec, inc, lat_s, lon_s):
    """
    由岩石剩磁方向 (偏角 dec, 倾角 inc) 和采样点坐标
    (lat_s, lon_s) 计算虚拟地磁极 (VGP) 位置。

    角度单位: 度
    使用地心偶极子公式 (e.g. Butler 1992, Tauxe 2010):
        p = arccot(tan(inc) / 2)
        lat_p = arcsin(sin(lat_s) cos(p) + cos(lat_s) sin(p) cos(dec))
        lon_p = lon_s + arcsin(sin(p) sin(dec) / cos(lat_p))
    并根据球面象限规则进行经度校正。

    Parameters
    ----------
    dec : float or array-like
        偏角 (declination), 度
    inc : float or array-like
        倾角 (inclination), 度
    lat_s : float or array-like
        采样点纬度, 度 (北正南负)
    lon_s : float or array-like
        采样点经度, 度 (东正西负)

    Returns
    -------
    lat_p : float or ndarray
        VGP 纬度, 度
    lon_p : float or ndarray
        VGP 经度, 度
    """
    dec_r = np.deg2rad(dec)
    inc_r = np.deg2rad(inc)
    lat_r = np.deg2rad(lat_s)
    lon_r = np.deg2rad(lon_s)

    # 1) 极距 p (采样点到 VGP 的角距离)
    p = np.arctan(2.0 / np.tan(inc_r))

    # 2) VGP 纬度
    sin_lat_p = (np.sin(lat_r) * np.cos(p)
                 + np.cos(lat_r) * np.sin(p) * np.cos(dec_r))
    lat_p = np.arcsin(sin_lat_p)

    # 3) VGP 经度 (含象限判断)
    # 球面三角正弦给出 sin(lon_p - lon_s)
    sin_dlon = np.sin(p) * np.sin(dec_r) / np.cos(lat_p)
    # 球面三角余弦给出 cos(lon_p - lon_s)
    cos_dlon = (np.cos(p) - np.sin(lat_r) * sin_lat_p) / (
        np.cos(lat_r) * np.cos(lat_p)
    )
    dlon = np.arctan2(sin_dlon, cos_dlon)
    lon_p = lon_r + dlon

    # 经度归一化到 [-180, 180]
    lon_p = np.mod(lon_p + np.pi, 2.0 * np.pi) - np.pi

    return np.rad2deg(lat_p), np.rad2deg(lon_p)


def vgp_to_dec_inc(lat_p, lon_p, lat_s, lon_s):
    """
    反向: 由 VGP 位置和采样点坐标计算偏角、倾角。
    用于对上述函数做交叉验证。
    """
    lat_p = np.deg2rad(lat_p)
    lon_p = np.deg2rad(lon_p)
    lat_s = np.deg2rad(lat_s)
    lon_s = np.deg2rad(lon_s)

    # 倾角: tan(I) = 2 tan(lambda) 不适用于非极坐标,
    # 通用公式: cot(p) = 0.5 tan(inc)  => tan(inc) = 2 cos(p)? 
    # 其实: tan(inc) = 2 / tan(p) ... but here we compute from pole:
    # cos(p) = sin(lat_s) sin(lat_p) + cos(lat_s) cos(lat_p) cos(DLon)
    cos_p = (np.sin(lat_s) * np.sin(lat_p)
             + np.cos(lat_s) * np.cos(lat_p) * np.cos(lon_p - lon_s))
    sin_p = np.sqrt(np.clip(1.0 - cos_p * cos_p, 0.0, 1.0))
    inc = np.arctan(2.0 * cos_p / sin_p)

    sin_dec = np.sin(lon_p - lon_s) * np.cos(lat_p) / sin_p
    cos_dec = (np.sin(lat_p) - np.sin(lat_s) * cos_p) / (
        np.cos(lat_s) * sin_p
    )
    dec = np.arctan2(sin_dec, cos_dec)
    dec = np.mod(dec, 2.0 * np.pi)
    return np.rad2deg(dec), np.rad2deg(inc)


if __name__ == "__main__":
    # 示例: 采样点位于 (30°N, 120°E), 剩磁方向 D=0°, I=60°
    D, I = 0.0, 60.0
    lat_s, lon_s = 30.0, 120.0
    lat_p, lon_p = dec_inc_to_vgp(D, I, lat_s, lon_s)
    print(f"D={D}, I={I}, site=({lat_s}, {lon_s})  =>  VGP=({lat_p:.4f}, {lon_p:.4f})")

    # 反算验证
    D2, I2 = vgp_to_dec_inc(lat_p, lon_p, lat_s, lon_s)
    print(f"反向计算: D'={D2:.4f}, I'={I2:.4f}  (应与 D={D}, I={I} 一致)")

    # 批量示例
    Ds = np.array([0.0, 90.0, 180.0, 270.0])
    Is = np.array([60.0, 45.0, 30.0, 15.0])
    lats, lons = dec_inc_to_vgp(Ds, Is, lat_s, lon_s)
    print("\n批量结果:")
    for d, i, lat, lon in zip(Ds, Is, lats, lons):
        print(f"  D={d:5.1f}  I={i:5.1f}  =>  lat_p={lat:9.4f}  lon_p={lon:9.4f}")
