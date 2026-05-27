import math
from typing import Optional, Dict, List, Tuple, Callable
import itertools


class CycloneSeparator:
    def __init__(
        self,
        diameter: float,
        inlet_height_ratio: Optional[float] = None,
        inlet_width_ratio: Optional[float] = None,
        vortex_finder_ratio: Optional[float] = None,
        cone_angle: Optional[float] = None,
        outlet_ratio: Optional[float] = None,
        fluid_density: float = 1.2,
        fluid_viscosity: float = 1.81e-5,
        particle_density: float = 2000.0,
    ):
        self.D = diameter
        self.a_ratio = inlet_height_ratio if inlet_height_ratio is not None else 0.5
        self.b_ratio = inlet_width_ratio if inlet_width_ratio is not None else 0.25
        self.De_ratio = vortex_finder_ratio if vortex_finder_ratio is not None else 0.5
        self.cone_angle = cone_angle if cone_angle is not None else 15.0
        self.B_ratio = outlet_ratio if outlet_ratio is not None else 0.25

        self.a = self.a_ratio * diameter
        self.b = self.b_ratio * diameter
        self.De = self.De_ratio * diameter
        self.B = self.B_ratio * diameter

        tan_theta = math.tan(math.radians(self.cone_angle))
        self.Hc = (self.D - self.B) / (2 * tan_theta) if tan_theta > 0 else 2.0 * diameter

        self.rho_f = fluid_density
        self.mu = fluid_viscosity
        self.rho_p = particle_density

    @classmethod
    def from_absolute_dimensions(
        cls,
        diameter: float,
        inlet_height: float,
        inlet_width: float,
        vortex_finder_diameter: float,
        cone_height: float,
        outlet_diameter: float,
        fluid_density: float = 1.2,
        fluid_viscosity: float = 1.81e-5,
        particle_density: float = 2000.0,
    ):
        inlet_height_ratio = inlet_height / diameter
        inlet_width_ratio = inlet_width / diameter
        vortex_finder_ratio = vortex_finder_diameter / diameter
        outlet_ratio = outlet_diameter / diameter

        delta_r = (diameter - outlet_diameter) / 2
        cone_angle = math.degrees(math.atan(delta_r / cone_height))

        return cls(
            diameter=diameter,
            inlet_height_ratio=inlet_height_ratio,
            inlet_width_ratio=inlet_width_ratio,
            vortex_finder_ratio=vortex_finder_ratio,
            cone_angle=cone_angle,
            outlet_ratio=outlet_ratio,
            fluid_density=fluid_density,
            fluid_viscosity=fluid_viscosity,
            particle_density=particle_density
        )

    def update_geometry(
        self,
        inlet_height_ratio: Optional[float] = None,
        inlet_width_ratio: Optional[float] = None,
        vortex_finder_ratio: Optional[float] = None,
        cone_angle: Optional[float] = None,
        outlet_ratio: Optional[float] = None,
    ):
        if inlet_height_ratio is not None:
            self.a_ratio = inlet_height_ratio
            self.a = inlet_height_ratio * self.D
        if inlet_width_ratio is not None:
            self.b_ratio = inlet_width_ratio
            self.b = inlet_width_ratio * self.D
        if vortex_finder_ratio is not None:
            self.De_ratio = vortex_finder_ratio
            self.De = vortex_finder_ratio * self.D
        if cone_angle is not None:
            self.cone_angle = cone_angle
            tan_theta = math.tan(math.radians(cone_angle))
            self.Hc = (self.D - self.B) / (2 * tan_theta) if tan_theta > 0 else 2.0 * self.D
        if outlet_ratio is not None:
            self.B_ratio = outlet_ratio
            self.B = outlet_ratio * self.D
            tan_theta = math.tan(math.radians(self.cone_angle))
            self.Hc = (self.D - self.B) / (2 * tan_theta) if tan_theta > 0 else 2.0 * self.D

    def calculate_inlet_velocity(self, flow_rate: float) -> float:
        inlet_area = self.a * self.b
        return flow_rate / inlet_area

    def lapple_model_d50(self, inlet_velocity: float) -> float:
        N_e = 5.0
        d50 = math.sqrt(
            (9 * self.mu * self.b) /
            (2 * math.pi * N_e * inlet_velocity * (self.rho_p - self.rho_f))
        )
        return d50

    def bartholomew_model_d50(self, inlet_velocity: float) -> float:
        N_e = 6.0
        d50 = math.sqrt(
            (9 * self.mu * self.b) /
            (2 * math.pi * N_e * inlet_velocity * (self.rho_p - self.rho_f))
        )
        return d50

    def muschelknautz_model_d50(self, inlet_velocity: float) -> float:
        N_e = 5.5
        d50 = math.sqrt(
            (9 * self.mu * self.b) /
            (2 * math.pi * N_e * inlet_velocity * (self.rho_p - self.rho_f))
        )
        return d50

    def mothes_loffler_model_d50(self, inlet_velocity: float) -> float:
        N_e = 5.5
        d50_stokes = math.sqrt(
            (9 * self.mu * self.b) /
            (2 * math.pi * N_e * inlet_velocity * (self.rho_p - self.rho_f))
        )
        Re = self.rho_f * inlet_velocity * self.b / self.mu
        turbulent_intensity = 0.16 * Re ** (-0.125)
        correction_factor = 1.0 + 1.5 * turbulent_intensity
        d50 = d50_stokes * correction_factor
        return d50

    def cfd_dem_corrected_d50(self, inlet_velocity: float) -> float:
        N_e = 5.2
        d50_stokes = math.sqrt(
            (9 * self.mu * self.b) /
            (2 * math.pi * N_e * inlet_velocity * (self.rho_p - self.rho_f))
        )
        Re = self.rho_f * inlet_velocity * self.D / self.mu
        turbulent_intensity = 0.16 * Re ** (-0.125)
        correction_factor = 1.0 + 2.5 * turbulent_intensity
        d50 = d50_stokes * correction_factor
        return d50

    def calculate_pressure_drop_simple(self, inlet_velocity: float) -> float:
        K = 16 * (self.a * self.b) / (self.De ** 2)
        delta_P = K * 0.5 * self.rho_f * (inlet_velocity ** 2)
        return delta_P

    def calculate_pressure_drop_shepherd_lapple(self, inlet_velocity: float) -> float:
        K_SL = 11.3 * (self.a * self.b) / (self.De ** 2) / (16 * 0.5 * 0.25 / (0.5 ** 2))
        K_SL = max(K_SL, 5.0)
        K_SL = min(K_SL, 20.0)
        delta_P = K_SL * self.rho_f * (inlet_velocity ** 2) / 2
        return delta_P

    def calculate_pressure_drop_stairmand(self, inlet_velocity: float) -> float:
        K = 32.0 * self.a * self.b / (self.De ** 2)
        delta_P = K * self.rho_f * (inlet_velocity ** 2) / 2
        return delta_P

    def calculate_pressure_drop_dirgo(self, inlet_velocity: float) -> float:
        K = 4.72 * (self.D ** 2) / (self.a * self.b)
        delta_P = K * self.rho_f * (inlet_velocity ** 2) / 2
        return delta_P

    def calculate_pressure_drop_casals(self, inlet_velocity: float) -> float:
        K = 20.0 * (self.a / self.D) * (self.b / self.D) ** 0.5 / (self.De / self.D) ** 2
        delta_P = K * self.rho_f * (inlet_velocity ** 2) / 2
        return delta_P

    def calculate_pressure_drop(
        self,
        inlet_velocity: float,
        model: str = 'stairmand'
    ) -> float:
        model = model.lower()
        if model == 'simple':
            return self.calculate_pressure_drop_simple(inlet_velocity)
        elif model == 'shepherd_lapple':
            return self.calculate_pressure_drop_shepherd_lapple(inlet_velocity)
        elif model == 'stairmand':
            return self.calculate_pressure_drop_stairmand(inlet_velocity)
        elif model == 'dirgo':
            return self.calculate_pressure_drop_dirgo(inlet_velocity)
        elif model == 'casals':
            return self.calculate_pressure_drop_casals(inlet_velocity)
        else:
            raise ValueError(f"Unknown pressure drop model: {model}")

    def _classical_grade_efficacy(self, particle_diameter: float, d50: float, exponent: float) -> float:
        if particle_diameter <= 0:
            return 0.0
        ratio = particle_diameter / d50
        efficacy = 1 - math.exp(-0.693 * (ratio ** exponent))
        return efficacy * 100

    def mothes_loffler_grade_efficacy(self, particle_diameter: float, d50: float, inlet_velocity: float) -> float:
        if particle_diameter <= 0:
            return 0.0
        ratio = particle_diameter / d50
        base_eff = 1 - math.exp(-0.693 * (ratio ** 2.0))
        Re = self.rho_f * inlet_velocity * self.b / self.mu
        turbulent_intensity = 0.16 * Re ** (-0.125)
        stokes = (self.rho_p - self.rho_f) * particle_diameter ** 2 * inlet_velocity / (18 * self.mu * self.D)
        if stokes < 0.01:
            correction = 0.3 + 0.7 * math.exp(-100 * stokes) * turbulent_intensity * 5
        else:
            correction = 1.0 - 0.5 * turbulent_intensity * math.exp(-5.0 * (stokes - 0.01))
        if correction < 0.1:
            correction = 0.1
        if correction > 1.0:
            correction = 1.0
        eta = base_eff * correction
        return eta * 100

    def leith_licht_grade_efficacy(self, particle_diameter: float, inlet_velocity: float) -> float:
        if particle_diameter <= 0:
            return 0.0
        n = 1.0 - (1 - 0.67 * self.D ** 0.14) * (inlet_velocity / 20) ** 0.3
        if n < 0.5:
            n = 0.5
        v_theta_i = inlet_velocity * (self.D + self.b) / (2 * self.D)
        r_c = self.De / 2
        tau = (self.rho_p - self.rho_f) * particle_diameter ** 2 / (18 * self.mu)
        G = 2.0 * tau * v_theta_i ** 2 / (r_c * inlet_velocity)
        eta = 1 - math.exp(-2 * (G * n) ** 0.5)
        return eta * 100

    def cfd_dem_grade_efficacy(self, particle_diameter: float, d50: float, inlet_velocity: float) -> float:
        if particle_diameter <= 0:
            return 0.0
        Re = self.rho_f * inlet_velocity * self.D / self.mu
        turbulent_intensity = 0.16 * Re ** (-0.125)
        stokes = (self.rho_p - self.rho_f) * particle_diameter ** 2 * inlet_velocity / (18 * self.mu * self.D)
        ratio = particle_diameter / d50
        base_eff = 1 - math.exp(-0.693 * (ratio ** 2.2))
        turb_correction = 1.0 - 0.6 * turbulent_intensity * math.exp(-5.0 * stokes)
        if turb_correction < 0.3:
            turb_correction = 0.3
        eta = base_eff * turb_correction
        return eta * 100

    def salcedo_grade_efficacy(self, particle_diameter: float, d50: float, inlet_velocity: float) -> float:
        if particle_diameter <= 0:
            return 0.0
        Re = self.rho_f * inlet_velocity * self.D / self.mu
        ratio = particle_diameter / d50
        if Re < 3000:
            exponent = 1.8
        elif Re < 10000:
            exponent = 2.0
        else:
            exponent = 2.3
        base_eff = 1 - math.exp(-0.693 * (ratio ** exponent))
        if particle_diameter < 2e-6:
            reduction_factor = 0.5 + 0.5 * math.exp(-particle_diameter / 0.3e-6)
            base_eff = base_eff * (1 - reduction_factor * 0.8)
        return base_eff * 100

    def grade_efficacy(
        self,
        particle_diameter: float,
        d50: float,
        model: str = 'lapple',
        inlet_velocity: Optional[float] = None
    ) -> float:
        model = model.lower()

        if model == 'lapple':
            return self._classical_grade_efficacy(particle_diameter, d50, 2.0)
        elif model == 'bartholomew':
            return self._classical_grade_efficacy(particle_diameter, d50, 2.5)
        elif model == 'muschelknautz':
            return self._classical_grade_efficacy(particle_diameter, d50, 3.0)
        elif model == 'mothes_loffler':
            if inlet_velocity is None:
                raise ValueError("inlet_velocity is required for Mothes-Loffler model")
            return self.mothes_loffler_grade_efficacy(particle_diameter, d50, inlet_velocity)
        elif model == 'leith_licht':
            if inlet_velocity is None:
                raise ValueError("inlet_velocity is required for Leith-Licht model")
            return self.leith_licht_grade_efficacy(particle_diameter, inlet_velocity)
        elif model == 'cfd_dem':
            if inlet_velocity is None:
                raise ValueError("inlet_velocity is required for CFD-DEM model")
            return self.cfd_dem_grade_efficacy(particle_diameter, d50, inlet_velocity)
        elif model == 'salcedo':
            if inlet_velocity is None:
                raise ValueError("inlet_velocity is required for Salcedo model")
            return self.salcedo_grade_efficacy(particle_diameter, d50, inlet_velocity)
        else:
            raise ValueError(f"Unknown model: {model}")

    def calculate_d50(self, inlet_velocity: float, model: str = 'lapple') -> float:
        model = model.lower()
        if model == 'lapple':
            return self.lapple_model_d50(inlet_velocity)
        elif model == 'bartholomew':
            return self.bartholomew_model_d50(inlet_velocity)
        elif model == 'muschelknautz':
            return self.muschelknautz_model_d50(inlet_velocity)
        elif model == 'mothes_loffler':
            return self.mothes_loffler_model_d50(inlet_velocity)
        elif model == 'leith_licht':
            return self.lapple_model_d50(inlet_velocity) * 0.95
        elif model == 'cfd_dem':
            return self.cfd_dem_corrected_d50(inlet_velocity)
        elif model == 'salcedo':
            return self.lapple_model_d50(inlet_velocity) * 1.05
        else:
            raise ValueError(f"Unknown model: {model}")

    def comprehensive_analysis(
        self,
        inlet_velocity: float,
        particle_diameters: List[float],
        model: str = 'lapple',
        pressure_model: str = 'shepherd_lapple'
    ) -> Dict:
        model = model.lower()
        d50 = self.calculate_d50(inlet_velocity, model)
        pressure_drop = self.calculate_pressure_drop(inlet_velocity, pressure_model)

        grade_efficacies = []
        for dp in particle_diameters:
            eff = self.grade_efficacy(dp, d50, model, inlet_velocity)
            grade_efficacies.append({'diameter': dp, 'efficacy': eff})

        return {
            'model': model,
            'd50': d50,
            'inlet_velocity': inlet_velocity,
            'pressure_drop': pressure_drop,
            'grade_efficacies': grade_efficacies,
            'geometry': {
                'D': self.D,
                'a': self.a,
                'b': self.b,
                'De': self.De,
                'Hc': self.Hc,
                'B': self.B,
                'cone_angle': self.cone_angle
            }
        }

    def compare_models(
        self,
        inlet_velocity: float,
        particle_diameters: List[float],
        models: Optional[List[str]] = None
    ) -> List[Dict]:
        if models is None:
            models = ['lapple', 'mothes_loffler', 'cfd_dem', 'leith_licht', 'salcedo']

        results = []
        for model in models:
            result = self.comprehensive_analysis(inlet_velocity, particle_diameters, model)
            results.append(result)
        return results

    def get_geometry_ratios(self) -> Dict:
        return {
            'a/D': self.a_ratio,
            'b/D': self.b_ratio,
            'De/D': self.De_ratio,
            'B/D': self.B_ratio,
            'cone_angle': self.cone_angle,
            'Hc/D': self.Hc / self.D
        }


class CycloneOptimizer:
    def __init__(
        self,
        base_diameter: float,
        fluid_density: float = 1.2,
        fluid_viscosity: float = 1.81e-5,
        particle_density: float = 2000.0,
    ):
        self.base_D = base_diameter
        self.rho_f = fluid_density
        self.mu = fluid_viscosity
        self.rho_p = particle_density

    def create_cyclone(
        self,
        a_ratio: float,
        b_ratio: float,
        De_ratio: float,
        cone_angle: float,
        B_ratio: float
    ) -> CycloneSeparator:
        return CycloneSeparator(
            diameter=self.base_D,
            inlet_height_ratio=a_ratio,
            inlet_width_ratio=b_ratio,
            vortex_finder_ratio=De_ratio,
            cone_angle=cone_angle,
            outlet_ratio=B_ratio,
            fluid_density=self.rho_f,
            fluid_viscosity=self.mu,
            particle_density=self.rho_p
        )

    def evaluate_design(
        self,
        params: Dict[str, float],
        inlet_velocity: float,
        d50_model: str = 'mothes_loffler',
        pressure_model: str = 'shepherd_lapple'
    ) -> Dict:
        cyclone = self.create_cyclone(
            a_ratio=params.get('a_ratio', 0.5),
            b_ratio=params.get('b_ratio', 0.25),
            De_ratio=params.get('De_ratio', 0.5),
            cone_angle=params.get('cone_angle', 15.0),
            B_ratio=params.get('B_ratio', 0.25)
        )

        d50 = cyclone.calculate_d50(inlet_velocity, d50_model)
        pressure_drop = cyclone.calculate_pressure_drop(inlet_velocity, pressure_model)

        return {
            'params': params,
            'd50': d50,
            'pressure_drop': pressure_drop,
            'd50_um': d50 * 1e6,
            'cyclone': cyclone
        }

    def full_factorial_design(
        self,
        param_ranges: Dict[str, Tuple[float, float, int]],
        inlet_velocity: float,
        d50_model: str = 'mothes_loffler'
    ) -> List[Dict]:
        param_names = list(param_ranges.keys())
        param_values = []
        for name, (start, end, n_points) in param_ranges.items():
            if n_points == 1:
                values = [start]
            else:
                step = (end - start) / (n_points - 1)
                values = [start + i * step for i in range(n_points)]
            param_values.append(values)

        results = []
        for combo in itertools.product(*param_values):
            params = dict(zip(param_names, combo))
            result = self.evaluate_design(params, inlet_velocity, d50_model)
            results.append(result)

        return results

    def response_surface_optimization(
        self,
        inlet_velocity: float,
        target_d50: Optional[float] = None,
        max_pressure_drop: Optional[float] = None,
        d50_model: str = 'mothes_loffler',
        pressure_model: str = 'shepherd_lapple'
    ) -> Dict:
        param_space = {
            'a_ratio': (0.35, 0.6, 4),
            'b_ratio': (0.15, 0.35, 4),
            'De_ratio': (0.35, 0.65, 4),
            'cone_angle': (10.0, 25.0, 4),
            'B_ratio': (0.15, 0.35, 4)
        }

        results = self.full_factorial_design(param_space, inlet_velocity, d50_model)

        for r in results:
            if target_d50 is not None:
                d50_deviation = abs(r['d50'] - target_d50) / target_d50
                r['objective'] = d50_deviation
            elif max_pressure_drop is not None:
                if r['pressure_drop'] > max_pressure_drop:
                    r['objective'] = float('inf')
                else:
                    r['objective'] = r['d50']
            else:
                r['objective'] = r['d50'] * (r['pressure_drop'] ** 0.3)

        valid_results = [r for r in results if r['objective'] != float('inf')]
        if not valid_results:
            raise ValueError("No valid designs found within constraints")

        optimal = min(valid_results, key=lambda x: x['objective'])

        return {
            'optimal_design': optimal,
            'all_designs': results,
            'n_designs_evaluated': len(results)
        }

    def multi_objective_optimization(
        self,
        inlet_velocity: float,
        d50_weight: float = 0.5,
        pressure_weight: float = 0.5,
        d50_model: str = 'mothes_loffler',
        pressure_model: str = 'shepherd_lapple'
    ) -> Dict:
        param_space = {
            'a_ratio': (0.35, 0.6, 4),
            'b_ratio': (0.15, 0.35, 4),
            'De_ratio': (0.35, 0.65, 4),
            'cone_angle': (10.0, 25.0, 4),
            'B_ratio': (0.15, 0.35, 4)
        }

        results = self.full_factorial_design(param_space, inlet_velocity, d50_model)

        d50_values = [r['d50'] for r in results]
        pressure_values = [r['pressure_drop'] for r in results]

        min_d50, max_d50 = min(d50_values), max(d50_values)
        min_p, max_p = min(pressure_values), max(pressure_values)

        for r in results:
            norm_d50 = (r['d50'] - min_d50) / (max_d50 - min_d50) if max_d50 > min_d50 else 0
            norm_p = (r['pressure_drop'] - min_p) / (max_p - min_p) if max_p > min_p else 0
            r['score'] = d50_weight * norm_d50 + pressure_weight * norm_p
            r['norm_d50'] = norm_d50
            r['norm_pressure'] = norm_p

        pareto_front = self._find_pareto_front(results)
        optimal = min(results, key=lambda x: x['score'])

        return {
            'optimal_design': optimal,
            'pareto_front': pareto_front,
            'all_designs': results,
            'min_d50': min_d50,
            'min_pressure': min_p,
            'max_d50': max_d50,
            'max_pressure': max_p
        }

    def _find_pareto_front(self, results: List[Dict]) -> List[Dict]:
        pareto = []
        for r in results:
            dominated = False
            for other in results:
                if r is other:
                    continue
                if (other['d50'] <= r['d50'] and other['pressure_drop'] <= r['pressure_drop'] and
                    (other['d50'] < r['d50'] or other['pressure_drop'] < r['pressure_drop'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)

        pareto.sort(key=lambda x: x['d50'])
        return pareto

    def single_parameter_sensitivity(
        self,
        param_name: str,
        param_range: Tuple[float, float, int],
        base_params: Dict[str, float],
        inlet_velocity: float,
        d50_model: str = 'mothes_loffler',
        pressure_model: str = 'stairmand'
    ) -> List[Dict]:
        results = []
        start, end, n_points = param_range

        for i in range(n_points):
            if n_points == 1:
                value = start
            else:
                value = start + i * (end - start) / (n_points - 1)

            params = base_params.copy()
            params[param_name] = value

            result = self.evaluate_design(params, inlet_velocity, d50_model, pressure_model)
            result['sensitivity_param'] = param_name
            result['param_value'] = value
            results.append(result)

        return results


def main():
    print("=" * 70)
    print("Cyclone Separator Design & Optimization Tool")
    print("=" * 70)

    try:
        D_input = input("\nEnter cyclone body diameter D (m, default 0.1): ").strip()
        D = float(D_input) if D_input else 0.1
    except ValueError:
        D = 0.1

    try:
        rho_p_input = input("Enter particle density (kg/m^3, default 2000): ").strip()
        rho_p = float(rho_p_input) if rho_p_input else 2000.0
    except ValueError:
        rho_p = 2000.0

    try:
        rho_f_input = input("Enter fluid density (kg/m^3, default 1.2): ").strip()
        rho_f = float(rho_f_input) if rho_f_input else 1.2
    except ValueError:
        rho_f = 1.2

    try:
        mu_input = input("Enter fluid viscosity (Pa.s, default 1.81e-5): ").strip()
        mu = float(mu_input) if mu_input else 1.81e-5
    except ValueError:
        mu = 1.81e-5

    cyclone = CycloneSeparator(
        diameter=D,
        fluid_density=rho_f,
        fluid_viscosity=mu,
        particle_density=rho_p
    )

    print("\n" + "=" * 70)
    print("Cyclone Geometric Parameters (Stairmand High Efficiency):")
    ratios = cyclone.get_geometry_ratios()
    for key, value in ratios.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 70)

    try:
        vel_input = input("\nEnter inlet velocity (m/s, default 15): ").strip()
        inlet_velocity = float(vel_input) if vel_input else 15.0
    except ValueError:
        inlet_velocity = 15.0

    print("\n" + "=" * 70)
    print("Pressure Drop Comparison (Different Models):")
    print("=" * 70)

    p_models = ['simple', 'shepherd_lapple', 'stairmand', 'dirgo']
    p_model_names = {
        'simple': 'Simple K-factor',
        'shepherd_lapple': 'Shepherd-Lapple (Industry Standard)',
        'stairmand': 'Stairmand',
        'dirgo': 'Dirgo'
    }

    print(f"\n{'Model':<35} {'Pressure Drop (Pa)':<20}")
    print("-" * 60)
    for model in p_models:
        dp = cyclone.calculate_pressure_drop(inlet_velocity, model)
        print(f"{p_model_names[model]:<35} {dp:<20.2f}")

    print("\n" + "=" * 70)
    print("d50 Calculation Results:")
    print("=" * 70)

    model_names = {
        'lapple': 'Lapple (Classical)',
        'bartholomew': 'Bartholomew',
        'muschelknautz': 'Muschelknautz',
        'mothes_loffler': 'Mothes-Loffler (Turbulence Corrected)',
        'cfd_dem': 'CFD-DEM Corrected'
    }

    all_models = ['lapple', 'bartholomew', 'muschelknautz', 'mothes_loffler', 'cfd_dem']

    d50_results = {}
    print(f"\n{'Model':<40} {'d50 (um)':<15}")
    print("-" * 60)
    for model in all_models:
        d50 = cyclone.calculate_d50(inlet_velocity, model)
        d50_results[model] = d50
        print(f"{model_names[model]:<40} {d50 * 1e6:<15.2f}")

    print("\n" + "=" * 70)
    print("Grade Efficiency Comparison (Fine Particle Region):")
    print("=" * 70)

    test_diameters = [0.1e-6, 0.5e-6, 1e-6, 2e-6, 5e-6, 10e-6]

    print(f"\n{'Diameter':<15} {'Lapple':<12} {'Mothes-Loffler':<18} {'CFD-DEM':<12}")
    print("-" * 60)

    for dp in test_diameters:
        d50_lapple = d50_results['lapple']
        d50_ml = d50_results['mothes_loffler']
        d50_cfd = d50_results['cfd_dem']

        eff_lapple = cyclone.grade_efficacy(dp, d50_lapple, 'lapple')
        eff_ml = cyclone.grade_efficacy(dp, d50_ml, 'mothes_loffler', inlet_velocity)
        eff_cfd = cyclone.grade_efficacy(dp, d50_cfd, 'cfd_dem', inlet_velocity)

        print(f"{dp * 1e6:<15.2f} {eff_lapple:<12.2f} {eff_ml:<18.2f} {eff_cfd:<12.2f}")

    print("\n" + "=" * 70)
    print("Geometry Optimization Example:")
    print("=" * 70)

    print("\nRunning Response Surface Optimization...")
    optimizer = CycloneOptimizer(
        base_diameter=D,
        fluid_density=rho_f,
        fluid_viscosity=mu,
        particle_density=rho_p
    )

    opt_result = optimizer.multi_objective_optimization(
        inlet_velocity=inlet_velocity,
        d50_weight=0.6,
        pressure_weight=0.4
    )

    opt = opt_result['optimal_design']
    print(f"\nOptimal Design Found:")
    print(f"  Geometry ratios:")
    for key, value in opt['params'].items():
        print(f"    {key}: {value:.4f}")
    print(f"  Performance:")
    print(f"    d50: {opt['d50_um']:.2f} um")
    print(f"    Pressure drop: {opt['pressure_drop']:.2f} Pa")
    print(f"  Pareto front size: {len(opt_result['pareto_front'])} designs")

    print("\n" + "=" * 70)
    print("Program finished")
    print("=" * 70)


if __name__ == "__main__":
    main()
