import numpy as np
from typing import Callable, Optional, Union, Tuple, List, Dict, Any


class NumericalDifferentiation:
    EPS = np.finfo(float).eps

    def __init__(self, h: float = None, auto_step: bool = True):
        self.h = h
        self.auto_step = auto_step

    def _optimal_h(self, x: float, method: str, order: int = 1) -> float:
        if order == 1:
            if method in ['forward', 'backward']:
                return np.sqrt(self.EPS) * (1.0 + abs(x))
            elif method == 'central':
                return np.cbrt(self.EPS) * (1.0 + abs(x))
        elif order == 2:
            if method in ['forward', 'backward']:
                return self.EPS ** (1.0 / 3.0) * (1.0 + abs(x))
            elif method == 'central':
                return self.EPS ** (1.0 / 4.0) * (1.0 + abs(x))
        elif order == 3:
            return self.EPS ** (1.0 / 4.0) * (1.0 + abs(x))
        else:
            raise ValueError(f"Order {order} not supported")
        raise ValueError("Method must be 'forward', 'backward', or 'central'")

    def _get_h(self, h: Optional[float], x: float, method: str, order: int = 1) -> float:
        if h is not None:
            return h
        if self.h is not None:
            return self.h
        if np.isscalar(x):
            return self._optimal_h(x, method, order)
        else:
            return self._optimal_h(np.mean(np.abs(x)), method, order)

    def _estimate_error_richardson(self, f: Callable[[float], float], x: float, h: float,
                                   method: str, order: int = 1) -> Tuple[float, float]:
        if order == 1:
            if method == 'forward':
                d1 = (f(x + h) - f(x)) / h
                d2 = (f(x + h / 2) - f(x)) / (h / 2)
                derivative = 2 * d2 - d1
                error = abs(d2 - d1)
                return derivative, error
            elif method == 'backward':
                d1 = (f(x) - f(x - h)) / h
                d2 = (f(x) - f(x - h / 2)) / (h / 2)
                derivative = 2 * d2 - d1
                error = abs(d2 - d1)
                return derivative, error
            elif method == 'central':
                d1 = (f(x + h) - f(x - h)) / (2 * h)
                d2 = (f(x + h / 2) - f(x - h / 2)) / (2 * (h / 2))
                derivative = (4 * d2 - d1) / 3
                error = abs(d2 - d1) / 3
                return derivative, error
        elif order == 2:
            if method == 'forward':
                d1 = (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)
                d2 = (f(x + h) - 2 * f(x + h / 2) + f(x)) / ((h / 2) ** 2)
                derivative = (4 * d2 - d1) / 3
                error = abs(d2 - d1) / 3
                return derivative, error
            elif method == 'backward':
                d1 = (f(x) - 2 * f(x - h) + f(x - 2 * h)) / (h ** 2)
                d2 = (f(x) - 2 * f(x - h / 2) + f(x - h)) / ((h / 2) ** 2)
                derivative = (4 * d2 - d1) / 3
                error = abs(d2 - d1) / 3
                return derivative, error
            elif method == 'central':
                d1 = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
                d2 = (f(x + h / 2) - 2 * f(x) + f(x - h / 2)) / ((h / 2) ** 2)
                derivative = (4 * d2 - d1) / 3
                error = abs(d2 - d1) / 3
                return derivative, error
        elif order == 3:
            d1 = (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
            d2 = (f(x + h) - 2 * f(x + h / 2) + 2 * f(x - h / 2) - f(x - h)) / (2 * (h / 2) ** 3)
            derivative = (8 * d2 - d1) / 7
            error = abs(d2 - d1) / 7
            return derivative, error
        raise ValueError(f"Order {order} or method {method} not supported")

    def _apply_to_array(self, func: Callable, *args, **kwargs) -> Any:
        x = args[1] if len(args) > 1 else kwargs.get('x')
        if x is None and len(args) > 2:
            x = args[2]
        if isinstance(x, np.ndarray) or (isinstance(x, (list, tuple)) and len(x) > 1):
            x_arr = np.asarray(x)
            estimate_error = kwargs.get('estimate_error', False)
            richardson = kwargs.get('richardson', False)
            if estimate_error or richardson:
                results = [func(args[0], xi, *args[2:], **kwargs) for xi in x_arr]
                derivatives = np.array([r[0] for r in results])
                errors = np.array([r[1] for r in results])
                return derivatives, errors
            else:
                return np.array([func(args[0], xi, *args[2:], **kwargs) for xi in x_arr])
        return func(*args, **kwargs)

    def forward_difference(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                          h: Optional[float] = None, estimate_error: bool = False,
                          richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.forward_difference, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'forward', order=1)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'forward', order=1)
        return (f(x + h) - f(x)) / h

    def backward_difference(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                           h: Optional[float] = None, estimate_error: bool = False,
                           richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.backward_difference, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'backward', order=1)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'backward', order=1)
        return (f(x) - f(x - h)) / h

    def central_difference(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                          h: Optional[float] = None, estimate_error: bool = False,
                          richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.central_difference, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'central', order=1)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'central', order=1)
        return (f(x + h) - f(x - h)) / (2 * h)

    def second_derivative_forward(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                                 h: Optional[float] = None, estimate_error: bool = False,
                                 richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.second_derivative_forward, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'forward', order=2)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'forward', order=2)
        return (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)

    def second_derivative_backward(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                                  h: Optional[float] = None, estimate_error: bool = False,
                                  richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.second_derivative_backward, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'backward', order=2)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'backward', order=2)
        return (f(x) - 2 * f(x - h) + f(x - 2 * h)) / (h ** 2)

    def second_derivative_central(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                                 h: Optional[float] = None, estimate_error: bool = False,
                                 richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.second_derivative_central, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'central', order=2)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'central', order=2)
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

    def third_derivative_central(self, f: Callable[[float], float], x: Union[float, np.ndarray],
                                h: Optional[float] = None, estimate_error: bool = False,
                                richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            return self._apply_to_array(self.third_derivative_central, f, x, h=h,
                                       estimate_error=estimate_error, richardson=richardson)
        h = self._get_h(h, x, 'central', order=3)
        if estimate_error or richardson:
            return self._estimate_error_richardson(f, x, h, 'central', order=3)
        return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)

    def compute_derivative(self, method: str, f: Union[Callable[[float], float], tuple],
                          x: Union[float, np.ndarray], order: int = 1,
                          h: Optional[float] = None, estimate_error: bool = False,
                          richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        method = method.lower()
        if order == 1:
            if isinstance(f, tuple):
                x_points, y_points = f
                if method == 'forward':
                    return self.forward_difference_discrete(x_points, y_points, x, h, estimate_error, richardson)
                elif method == 'backward':
                    return self.backward_difference_discrete(x_points, y_points, x, h, estimate_error, richardson)
                elif method == 'central':
                    return self.central_difference_discrete(x_points, y_points, x, h, estimate_error, richardson)
                else:
                    raise ValueError("Method must be 'forward', 'backward', or 'central'")
            else:
                if method == 'forward':
                    return self.forward_difference(f, x, h, estimate_error, richardson)
                elif method == 'backward':
                    return self.backward_difference(f, x, h, estimate_error, richardson)
                elif method == 'central':
                    return self.central_difference(f, x, h, estimate_error, richardson)
                else:
                    raise ValueError("Method must be 'forward', 'backward', or 'central'")
        elif order == 2:
            if isinstance(f, tuple):
                raise NotImplementedError("Second derivative for discrete points not implemented yet")
            if method == 'forward':
                return self.second_derivative_forward(f, x, h, estimate_error, richardson)
            elif method == 'backward':
                return self.second_derivative_backward(f, x, h, estimate_error, richardson)
            elif method == 'central':
                return self.second_derivative_central(f, x, h, estimate_error, richardson)
            else:
                raise ValueError("Method must be 'forward', 'backward', or 'central'")
        elif order == 3:
            if isinstance(f, tuple):
                raise NotImplementedError("Third derivative for discrete points not implemented yet")
            if method == 'central':
                return self.third_derivative_central(f, x, h, estimate_error, richardson)
            else:
                raise ValueError("Third derivative only supports 'central' method")
        else:
            raise ValueError(f"Order {order} not supported (use 1, 2, or 3)")

    def _estimate_error_discrete(self, y_points: np.ndarray, idx: int, method: str, dx: float) -> Tuple[float, float]:
        if method == 'forward':
            d1 = (y_points[idx + 1] - y_points[idx]) / dx
            if idx + 2 < len(y_points):
                d2 = (y_points[idx + 2] - y_points[idx]) / (2 * dx)
                derivative = 2 * d1 - d2
                error = abs(d1 - d2)
                return derivative, error
            return d1, abs(d1) * self.EPS * 1e6
        elif method == 'backward':
            d1 = (y_points[idx] - y_points[idx - 1]) / dx
            if idx - 2 >= 0:
                d2 = (y_points[idx] - y_points[idx - 2]) / (2 * dx)
                derivative = 2 * d1 - d2
                error = abs(d1 - d2)
                return derivative, error
            return d1, abs(d1) * self.EPS * 1e6
        elif method == 'central':
            d1 = (y_points[idx + 1] - y_points[idx - 1]) / (2 * dx)
            if idx - 2 >= 0 and idx + 2 < len(y_points):
                d2 = (y_points[idx + 2] - y_points[idx - 2]) / (4 * dx)
                derivative = (4 * d1 - d2) / 3
                error = abs(d1 - d2) / 3
                return derivative, error
            return d1, abs(d1) * self.EPS * 1e6
        else:
            raise ValueError("Method must be 'forward', 'backward', or 'central'")

    def forward_difference_discrete(self, x_points: np.ndarray, y_points: np.ndarray,
                                   x: Union[float, np.ndarray], h: Optional[float] = None,
                                   estimate_error: bool = False,
                                   richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            x_arr = np.asarray(x)
            if estimate_error or richardson:
                results = [self.forward_difference_discrete(x_points, y_points, xi, h, estimate_error, richardson) for xi in x_arr]
                derivatives = np.array([r[0] for r in results])
                errors = np.array([r[1] for r in results])
                return derivatives, errors
            else:
                return np.array([self.forward_difference_discrete(x_points, y_points, xi, h) for xi in x_arr])
        idx = np.argmin(np.abs(x_points - x))
        if idx >= len(x_points) - 1:
            raise ValueError("x is too close to the end of the discrete points for forward difference")
        dx = x_points[idx + 1] - x_points[idx]
        if estimate_error or richardson:
            return self._estimate_error_discrete(y_points, idx, 'forward', dx)
        return (y_points[idx + 1] - y_points[idx]) / dx

    def backward_difference_discrete(self, x_points: np.ndarray, y_points: np.ndarray,
                                    x: Union[float, np.ndarray], h: Optional[float] = None,
                                    estimate_error: bool = False,
                                    richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            x_arr = np.asarray(x)
            if estimate_error or richardson:
                results = [self.backward_difference_discrete(x_points, y_points, xi, h, estimate_error, richardson) for xi in x_arr]
                derivatives = np.array([r[0] for r in results])
                errors = np.array([r[1] for r in results])
                return derivatives, errors
            else:
                return np.array([self.backward_difference_discrete(x_points, y_points, xi, h) for xi in x_arr])
        idx = np.argmin(np.abs(x_points - x))
        if idx <= 0:
            raise ValueError("x is too close to the start of the discrete points for backward difference")
        dx = x_points[idx] - x_points[idx - 1]
        if estimate_error or richardson:
            return self._estimate_error_discrete(y_points, idx, 'backward', dx)
        return (y_points[idx] - y_points[idx - 1]) / dx

    def central_difference_discrete(self, x_points: np.ndarray, y_points: np.ndarray,
                                   x: Union[float, np.ndarray], h: Optional[float] = None,
                                   estimate_error: bool = False,
                                   richardson: bool = False) -> Union[float, np.ndarray, Tuple]:
        if not np.isscalar(x):
            x_arr = np.asarray(x)
            if estimate_error or richardson:
                results = [self.central_difference_discrete(x_points, y_points, xi, h, estimate_error, richardson) for xi in x_arr]
                derivatives = np.array([r[0] for r in results])
                errors = np.array([r[1] for r in results])
                return derivatives, errors
            else:
                return np.array([self.central_difference_discrete(x_points, y_points, xi, h) for xi in x_arr])
        idx = np.argmin(np.abs(x_points - x))
        if idx <= 0 or idx >= len(x_points) - 1:
            raise ValueError("x is too close to the boundary for central difference")
        dx = x_points[idx + 1] - x_points[idx]
        if estimate_error or richardson:
            return self._estimate_error_discrete(y_points, idx, 'central', dx)
        return (y_points[idx + 1] - y_points[idx - 1]) / (x_points[idx + 1] - x_points[idx - 1])

    def find_optimal_step(self, f: Callable[[float], float], x: float, method: str,
                         order: int = 1, h_start: float = 1e-2,
                         h_min: float = 1e-12) -> Tuple[float, float, float]:
        method = method.lower()
        h = h_start
        hs = []
        derivatives = []
        errors = []

        while h >= h_min:
            if order == 1:
                if method == 'forward':
                    d = (f(x + h) - f(x)) / h
                elif method == 'backward':
                    d = (f(x) - f(x - h)) / h
                elif method == 'central':
                    d = (f(x + h) - f(x - h)) / (2 * h)
                else:
                    raise ValueError("Method must be 'forward', 'backward', or 'central'")
            elif order == 2:
                if method == 'forward':
                    d = (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)
                elif method == 'backward':
                    d = (f(x) - 2 * f(x - h) + f(x - 2 * h)) / (h ** 2)
                elif method == 'central':
                    d = (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)
                else:
                    raise ValueError("Method must be 'forward', 'backward', or 'central'")
            elif order == 3:
                if method == 'central':
                    d = (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
                else:
                    raise ValueError("Third derivative only supports 'central' method")
            else:
                raise ValueError(f"Order {order} not supported")

            hs.append(h)
            derivatives.append(d)
            if len(derivatives) >= 2:
                errors.append(abs(derivatives[-1] - derivatives[-2]))
            else:
                errors.append(np.inf)
            h /= 2

        if len(errors) > 0 and np.min(errors) < np.inf:
            best_idx = np.argmin(errors)
            return hs[best_idx], derivatives[best_idx], errors[best_idx]
        else:
            best_idx = 0
            return hs[best_idx], derivatives[best_idx], errors[best_idx]

    def compare_methods(self, f: Callable[[float], float], x: float,
                       df_exact: Optional[Callable[[float], float]] = None,
                       d2f_exact: Optional[Callable[[float], float]] = None,
                       d3f_exact: Optional[Callable[[float], float]] = None,
                       h_values: Optional[List[float]] = None) -> Dict[str, Any]:
        if h_values is None:
            h_values = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]

        results = {
            'x': x,
            'h_values': h_values,
            'first_order': {},
            'second_order': {},
            'third_order': {},
            'auto_step': {}
        }

        exact1 = df_exact(x) if df_exact else None
        exact2 = d2f_exact(x) if d2f_exact else None
        exact3 = d3f_exact(x) if d3f_exact else None

        for method in ['forward', 'backward', 'central']:
            results['first_order'][method] = {'h': [], 'value': [], 'error': []}
            for h in h_values:
                d = self.compute_derivative(method, f, x, order=1, h=h)
                err = abs(d - exact1) if exact1 is not None else None
                results['first_order'][method]['h'].append(h)
                results['first_order'][method]['value'].append(d)
                results['first_order'][method]['error'].append(err)

            results['second_order'][method] = {'h': [], 'value': [], 'error': []}
            for h in h_values:
                d = self.compute_derivative(method, f, x, order=2, h=h)
                err = abs(d - exact2) if exact2 is not None else None
                results['second_order'][method]['h'].append(h)
                results['second_order'][method]['value'].append(d)
                results['second_order'][method]['error'].append(err)

            d_auto, err_auto = self.compute_derivative(method, f, x, order=1, estimate_error=True)
            err_true = abs(d_auto - exact1) if exact1 is not None else None
            results['auto_step'][method] = {
                'value': d_auto,
                'estimated_error': err_auto,
                'true_error': err_true
            }

        results['third_order']['central'] = {'h': [], 'value': [], 'error': []}
        for h in h_values:
            d = self.compute_derivative('central', f, x, order=3, h=h)
            err = abs(d - exact3) if exact3 is not None else None
            results['third_order']['central']['h'].append(h)
            results['third_order']['central']['value'].append(d)
            results['third_order']['central']['error'].append(err)

        return results

    def stability_analysis(self, f: Callable[[float], float], x: float,
                          order: int = 1, method: str = 'central',
                          h_start: float = 1e-1, h_end: float = 1e-12,
                          n_steps: int = 50) -> Dict[str, np.ndarray]:
        method = method.lower()
        hs = np.logspace(np.log10(h_start), np.log10(h_end), n_steps)
        derivatives = []
        errors = []

        for h in hs:
            d = self.compute_derivative(method, f, x, order=order, h=h)
            derivatives.append(d)

        derivatives = np.array(derivatives)
        for i in range(len(derivatives)):
            if i > 0:
                errors.append(abs(derivatives[i] - derivatives[i - 1]))
            else:
                errors.append(np.inf)

        return {
            'h': hs,
            'derivatives': derivatives,
            'step_errors': np.array(errors)
        }

    def print_comparison(self, results: Dict[str, Any]) -> None:
        print("=" * 80)
        print("数值微分方法精度对比分析")
        print("=" * 80)
        print(f"\n计算点 x = {results['x']}")

        for order_name, order_key in [('一阶导数', 'first_order'), ('二阶导数', 'second_order')]:
            print(f"\n{'=' * 80}")
            print(f"{order_name}")
            print("=" * 80)

            methods = ['forward', 'backward', 'central'] if order_key != 'third_order' else ['central']

            header = f"{'步长 h':>12}"
            for method in methods:
                header += f"  {method:>14} (误差)"
            print(header)
            print("-" * 80)

            for i, h in enumerate(results['h_values']):
                line = f"{h:>12.2e}"
                for method in methods:
                    val = results[order_key][method]['value'][i]
                    err = results[order_key][method]['error'][i]
                    err_str = f"{err:.2e}" if err is not None else "N/A"
                    line += f"  {val:>14.10f} ({err_str:>8})"
                print(line)

        print(f"\n{'=' * 80}")
        print("三阶导数 (仅中心差分)")
        print("=" * 80)
        header = f"{'步长 h':>12}  {'central':>14} (误差)"
        print(header)
        print("-" * 80)
        for i, h in enumerate(results['h_values']):
            val = results['third_order']['central']['value'][i]
            err = results['third_order']['central']['error'][i]
            err_str = f"{err:.2e}" if err is not None else "N/A"
            print(f"{h:>12.2e}  {val:>14.10f} ({err_str:>8})")

        if 'auto_step' in results:
            print(f"\n{'=' * 80}")
            print("自动步长结果 (一阶导数)")
            print("=" * 80)
            header = f"{'方法':>12} {'导数近似':>18} {'估计误差':>14} {'真实误差':>12}"
            print(header)
            print("-" * 60)
            for method in ['forward', 'backward', 'central']:
                auto_data = results['auto_step'][method]
                val = auto_data['value']
                est_err = auto_data['estimated_error']
                true_err = auto_data['true_error']
                true_err_str = f"{true_err:.2e}" if true_err is not None else "N/A"
                print(f"{method:>12} {val:>18.10f} {est_err:>14.2e} {true_err_str:>12}")


if __name__ == "__main__":
    nd = NumericalDifferentiation(auto_step=True)

    def f(x):
        return np.sin(x) + x ** 3

    def df_exact(x):
        return np.cos(x) + 3 * x ** 2

    def d2f_exact(x):
        return -np.sin(x) + 6 * x

    def d3f_exact(x):
        return -np.cos(x) + 6

    x_test = 1.0

    print("=" * 80)
    print("数值微分 - 高阶导数、批量计算与精度对比")
    print("=" * 80)
    print(f"\n测试函数: f(x) = sin(x) + x^3")
    print(f"计算点: x = {x_test}")
    print(f"一阶导数精确值: f'({x_test}) = {df_exact(x_test):.12f}")
    print(f"二阶导数精确值: f''({x_test}) = {d2f_exact(x_test):.12f}")
    print(f"三阶导数精确值: f'''({x_test}) = {d3f_exact(x_test):.12f}")

    print("\n" + "=" * 80)
    print("1. 二阶导数计算")
    print("=" * 80)

    print("\n--- 二阶导数 (中心差分) ---")
    d2_central, err_est = nd.second_derivative_central(f, x_test, estimate_error=True)
    true_err = abs(d2_central - d2f_exact(x_test))
    print(f"自动步长: {d2_central:.12f}")
    print(f"估计误差: {err_est:.2e}, 真实误差: {true_err:.2e}")

    d2_rich, err_rich = nd.second_derivative_central(f, x_test, richardson=True)
    true_err_rich = abs(d2_rich - d2f_exact(x_test))
    print(f"\nRichardson 外推: {d2_rich:.12f}")
    print(f"估计误差: {err_rich:.2e}, 真实误差: {true_err_rich:.2e}")

    print("\n" + "=" * 80)
    print("2. 三阶导数计算")
    print("=" * 80)

    d3_central, err_est = nd.third_derivative_central(f, x_test, estimate_error=True)
    true_err = abs(d3_central - d3f_exact(x_test))
    print(f"自动步长: {d3_central:.12f}")
    print(f"估计误差: {err_est:.2e}, 真实误差: {true_err:.2e}")

    d3_rich, err_rich = nd.third_derivative_central(f, x_test, richardson=True)
    true_err_rich = abs(d3_rich - d3f_exact(x_test))
    print(f"\nRichardson 外推: {d3_rich:.12f}")
    print(f"估计误差: {err_rich:.2e}, 真实误差: {true_err_rich:.2e}")

    print("\n" + "=" * 80)
    print("3. 批量计算 (数组输入)")
    print("=" * 80)

    x_array = np.linspace(0, np.pi, 5)
    print(f"\n计算点: {x_array}")

    d1_batch, err1_batch = nd.central_difference(f, x_array, estimate_error=True)
    d2_batch, err2_batch = nd.second_derivative_central(f, x_array, estimate_error=True)

    print(f"\n--- 一阶导数 ---")
    for xi, di, ei in zip(x_array, d1_batch, err1_batch):
        exact = df_exact(xi)
        true_err = abs(di - exact)
        print(f"x = {xi:.4f}: {di:.10f} (估计误差: {ei:.2e}, 真实误差: {true_err:.2e})")

    print(f"\n--- 二阶导数 ---")
    for xi, di, ei in zip(x_array, d2_batch, err2_batch):
        exact = d2f_exact(xi)
        true_err = abs(di - exact)
        print(f"x = {xi:.4f}: {di:.10f} (估计误差: {ei:.2e}, 真实误差: {true_err:.2e})")

    print("\n" + "=" * 80)
    print("4. 方法精度对比")
    print("=" * 80)

    comparison = nd.compare_methods(f, x_test, df_exact, d2f_exact, d3f_exact)
    nd.print_comparison(comparison)

    print("\n" + "=" * 80)
    print("5. 稳定性分析 (观察误差随步长变化)")
    print("=" * 80)

    stability = nd.stability_analysis(f, x_test, order=1, method='central')
    min_err_idx = np.argmin(stability['step_errors'][1:]) + 1
    print(f"\n中心差分一阶导数:")
    print(f"最优步长 (最小相邻差): {stability['h'][min_err_idx]:.2e}")
    print(f"对应导数值: {stability['derivatives'][min_err_idx]:.12f}")
    print(f"精确值: {df_exact(x_test):.12f}")
    print(f"真实误差: {abs(stability['derivatives'][min_err_idx] - df_exact(x_test)):.2e}")

    print("\n" + "=" * 80)
    print("6. 使用示例")
    print("=" * 80)
    print("""
# 高阶导数
d2 = nd.second_derivative_central(f, x)                # 二阶导数
d2, err = nd.second_derivative_central(f, x, estimate_error=True)  # 带误差
d3, err = nd.third_derivative_central(f, x, richardson=True)       # 三阶+Richardson

# 批量计算
x_array = np.linspace(0, 10, 100)
d1_array = nd.central_difference(f, x_array)            # 批量一阶导数
d2_array, err_array = nd.second_derivative_central(f, x_array, estimate_error=True)

# 统一接口
d1 = nd.compute_derivative('central', f, x, order=1)    # 一阶
d2 = nd.compute_derivative('central', f, x, order=2)    # 二阶
d3 = nd.compute_derivative('central', f, x, order=3)    # 三阶

# 精度对比
comparison = nd.compare_methods(f, x, df_exact, d2f_exact, d3f_exact)
nd.print_comparison(comparison)

# 稳定性分析
stability = nd.stability_analysis(f, x, order=1, method='central')
""")
