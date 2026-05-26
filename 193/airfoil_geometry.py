import numpy as np


class AirfoilGeometry:
    def __init__(self, n_points=100):
        self.n_points = n_points
        
    def generate_naca4(self, naca='0012', x=None):
        if x is None:
            beta = np.linspace(0, np.pi, self.n_points)
            x = (1 - np.cos(beta)) / 2
        
        m = int(naca[0]) / 100.0
        p = int(naca[1]) / 10.0
        t = int(naca[2:4]) / 100.0
        
        y_t = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                       0.2843 * x**3 - 0.1015 * x**4)
        
        if p == 0 or m == 0:
            y_c = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            y_c = np.where(x < p,
                          m / p**2 * (2 * p * x - x**2),
                          m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
            dyc_dx = np.where(x < p,
                             2 * m / p**2 * (p - x),
                             2 * m / (1 - p)**2 * (p - x))
        
        theta = np.arctan(dyc_dx)
        
        x_u = x - y_t * np.sin(theta)
        y_u = y_c + y_t * np.cos(theta)
        x_l = x + y_t * np.sin(theta)
        y_l = y_c - y_t * np.cos(theta)
        
        return x_u, y_u, x_l, y_l
    
    def apply_camber_deformation(self, x, y, camber_amplitude, camber_frequency, time):
        deformation = camber_amplitude * np.sin(2 * np.pi * camber_frequency * x) * np.sin(time)
        return x, y + deformation
    
    def apply_pitch_motion(self, x, y, alpha_mean, alpha_amp, pitch_frequency, time, pivot=0.25):
        alpha = alpha_mean + alpha_amp * np.sin(pitch_frequency * time)
        alpha_rad = np.radians(alpha)
        
        dx = x - pivot
        x_rot = pivot + dx * np.cos(alpha_rad) - y * np.sin(alpha_rad)
        y_rot = y * np.cos(alpha_rad) + dx * np.sin(alpha_rad)
        
        return x_rot, y_rot, alpha_rad
    
    def get_panel_geometry(self, x_u, y_u, x_l, y_l):
        x = np.concatenate([x_l, x_u[-2::-1]])
        y = np.concatenate([y_l, y_u[-2::-1]])
        
        n_panels = len(x) - 1
        x_mid = np.zeros(n_panels)
        y_mid = np.zeros(n_panels)
        length = np.zeros(n_panels)
        nx = np.zeros(n_panels)
        ny = np.zeros(n_panels)
        tx = np.zeros(n_panels)
        ty = np.zeros(n_panels)
        
        for i in range(n_panels):
            x_mid[i] = 0.5 * (x[i] + x[i+1])
            y_mid[i] = 0.5 * (y[i] + y[i+1])
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            length[i] = np.sqrt(dx**2 + dy**2)
            nx[i] = -dy / length[i]
            ny[i] = dx / length[i]
            tx[i] = dx / length[i]
            ty[i] = dy / length[i]
        
        return x, y, x_mid, y_mid, length, nx, ny, tx, ty


def test_geometry():
    geom = AirfoilGeometry(n_points=50)
    x_u, y_u, x_l, y_l = geom.generate_naca4('0012')
    
    time = 0.5
    x_u_def = geom.apply_camber_deformation(x_u, y_u, 0.02, 2.0, time)[0]
    y_u_def = geom.apply_camber_deformation(x_u, y_u, 0.02, 2.0, time)[1]
    
    print("Geometry generation test passed!")
    print(f"Upper surface points: {len(x_u)}")
    print(f"Max thickness: {np.max(y_u - y_l):.4f}")


if __name__ == "__main__":
    test_geometry()
