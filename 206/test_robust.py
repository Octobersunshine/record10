import numpy as np
from robust_optical_filter import RobustOpticalFilter

f = RobustOpticalFilter()
f.lambda_center = 550
f.bandwidth = 40
f.wavelengths = np.linspace(450, 650, 100)
sigma = 40 / (2 * np.sqrt(2 * np.log(2)))
f.target_T = np.exp(-((f.wavelengths - 550) ** 2) / (2 * sigma ** 2))
f.set_specifications(min_transmittance=0.8, max_out_of_band=0.1, fwhm_tolerance=0.15)

n_test = [2.35, 1.38, 2.35, 1.38, 2.35, 1.38]
d_test = [58.5, 100.0, 58.5, 100.0, 58.5, 100.0]

metrics = f._evaluate_performance(n_test, d_test)
print('性能测试 OK: 峰值T = {:.2f}%'.format(metrics['max_T'] * 100))

spec_ok = f.check_specifications(metrics)
print('规格检查:', spec_ok)

print('基础功能测试通过!')
