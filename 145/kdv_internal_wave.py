import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')


class KdVSolver:
    def __init__(self, L=50, N=1024, dt=0.001, T_max=10):
        self.L = L
        self.N = N
        self.dt = dt
        self.T_max = T_max
        self.x = np.linspace(-L/2, L/2, N)
        self.k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
        self.k2 = self.k ** 2
        self.k3 = self.k ** 3
        
    def soliton_solution(self, x, x0=0, c=1, A=1):
        xi = np.sqrt(c/12) * (x - x0 - c * 0)
        return A * c / 2 * (1 / np.cosh(xi)) ** 2
    
    def gaussian_wave(self, x, x0=0, sigma=2, amp=1):
        return amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    
    def terrain_function(self, x, terrain_type='flat', **kwargs):
        if terrain_type == 'flat':
            return np.ones_like(x)
        elif terrain_type == 'shelf':
            h1 = kwargs.get('h1', 1.0)
            h2 = kwargs.get('h2', 0.5)
            x_trans = kwargs.get('x_trans', 0)
            width = kwargs.get('width', 5)
            transition = 0.5 * (1 + np.tanh((x - x_trans) / width))
            return h1 * (1 - transition) + h2 * transition
        elif terrain_type == 'ridge':
            h0 = kwargs.get('h0', 1.0)
            height = kwargs.get('height', 0.3)
            x0 = kwargs.get('x0', 0)
            width = kwargs.get('width', 3)
            return h0 - height * np.exp(-((x - x0) ** 2) / (2 * width ** 2))
        elif terrain_type == 'trench':
            h0 = kwargs.get('h0', 1.0)
            depth = kwargs.get('depth', 0.3)
            x0 = kwargs.get('x0', 0)
            width = kwargs.get('width', 3)
            return h0 + depth * np.exp(-((x - x0) ** 2) / (2 * width ** 2))
        else:
            return np.ones_like(x)
    
    def rhs_kdv(self, u_hat, t, h_x=None):
        u = np.fft.ifft(u_hat).real
        
        if h_x is not None:
            h = h_x
            h_x_deriv = np.gradient(h, self.x)
            nonlinear_term = -6 * (1/h) * u * np.gradient(u, self.x) - (h_x_deriv/h) * u
        else:
            nonlinear_term = -6 * u * np.gradient(u, self.x)
        
        nonlinear_hat = np.fft.fft(nonlinear_term)
        linear_term = -1j * self.k3 * u_hat
        
        return nonlinear_hat + linear_term
    
    def solve(self, u0, terrain_type='flat', **terrain_kwargs):
        h_x = self.terrain_function(self.x, terrain_type, **terrain_kwargs)
        
        num_steps = int(self.T_max / self.dt)
        u = u0.copy()
        u_hat = np.fft.fft(u)
        
        u_history = np.zeros((num_steps + 1, self.N))
        u_history[0] = u
        
        for i in range(num_steps):
            k1 = self.rhs_kdv(u_hat, i * self.dt, h_x)
            k2 = self.rhs_kdv(u_hat + 0.5 * self.dt * k1, i * self.dt + 0.5 * self.dt, h_x)
            k3 = self.rhs_kdv(u_hat + 0.5 * self.dt * k2, i * self.dt + 0.5 * self.dt, h_x)
            k4 = self.rhs_kdv(u_hat + self.dt * k3, i * self.dt + self.dt, h_x)
            
            u_hat = u_hat + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
            u = np.fft.ifft(u_hat).real
            u_history[i + 1] = u
        
        return u_history, h_x
    
    def plot_evolution(self, u_history, h_x=None, save_path='kdv_evolution.png'):
        t_indices = np.linspace(0, len(u_history) - 1, 5, dtype=int)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for idx in t_indices:
            t = idx * self.dt
            axes[0].plot(self.x, u_history[idx], label=f't={t:.2f}')
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('KdV Equation - Internal Wave Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if h_x is not None:
            axes[1].plot(self.x, h_x, 'k-', linewidth=2)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('Water Depth h(x)')
            axes[1].set_title('Terrain Profile')
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def plot_spacetime(self, u_history, save_path='kdv_spacetime.png'):
        t = np.arange(len(u_history)) * self.dt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.pcolormesh(self.x, t, u_history, cmap='RdBu_r', 
                           shading='auto', vmin=-np.max(np.abs(u_history)), 
                           vmax=np.max(np.abs(u_history)))
        
        ax.set_xlabel('x')
        ax.set_ylabel('Time')
        ax.set_title('Spacetime Evolution of Internal Waves')
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def create_animation(self, u_history, h_x=None, save_path='kdv_animation.gif', fps=30):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        line, = axes[0].plot(self.x, u_history[0], 'b-', linewidth=2)
        axes[0].set_xlim(self.x[0], self.x[-1])
        axes[0].set_ylim(1.2 * np.min(u_history), 1.2 * np.max(u_history))
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Internal Wave Propagation')
        axes[0].grid(True, alpha=0.3)
        
        if h_x is not None:
            terrain_line, = axes[1].plot(self.x, h_x, 'k-', linewidth=2)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('Water Depth h(x)')
            axes[1].set_title('Terrain Profile')
            axes[1].grid(True, alpha=0.3)
            axes[1].invert_yaxis()
        else:
            axes[1].axis('off')
        
        time_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes)
        
        def update(frame):
            line.set_ydata(u_history[frame])
            time_text.set_text(f'Time = {frame * self.dt:.3f}')
            return line, time_text
        
        anim = FuncAnimation(fig, update, frames=len(u_history), 
                             interval=1000/fps, blit=True)
        
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()


def main():
    print("Initializing KdV Solver for Internal Wave Simulation...")
    
    solver = KdVSolver(L=100, N=1024, dt=0.001, T_max=5)
    
    print("Creating initial condition: Single Soliton")
    u0 = solver.soliton_solution(solver.x, x0=-30, c=2, A=1)
    
    print("Solving with continental shelf terrain...")
    u_history_shelf, h_shelf = solver.solve(
        u0, 
        terrain_type='shelf',
        h1=1.0,
        h2=0.4,
        x_trans=10,
        width=3
    )
    
    solver.plot_evolution(u_history_shelf, h_shelf, 'shelf_evolution.png')
    solver.plot_spacetime(u_history_shelf, 'shelf_spacetime.png')
    print("Saved shelf simulation results!")
    
    print("Solving with submarine ridge...")
    u_history_ridge, h_ridge = solver.solve(
        u0,
        terrain_type='ridge',
        h0=1.0,
        height=0.5,
        x0=0,
        width=5
    )
    
    solver.plot_evolution(u_history_ridge, h_ridge, 'ridge_evolution.png')
    solver.plot_spacetime(u_history_ridge, 'ridge_spacetime.png')
    print("Saved ridge simulation results!")
    
    print("Solving with trench terrain...")
    u_history_trench, h_trench = solver.solve(
        u0,
        terrain_type='trench',
        h0=1.0,
        depth=0.4,
        x0=0,
        width=5
    )
    
    solver.plot_evolution(u_history_trench, h_trench, 'trench_evolution.png')
    solver.plot_spacetime(u_history_trench, 'trench_spacetime.png')
    print("Saved trench simulation results!")
    
    print("Creating animation for shelf case...")
    step = max(1, len(u_history_shelf) // 100)
    solver.create_animation(
        u_history_shelf[::step], 
        h_shelf, 
        'shelf_animation.gif',
        fps=20
    )
    print("Animation saved!")
    
    print("\nSimulation completed successfully!")
    print("Generated files:")
    print("- shelf_evolution.png, shelf_spacetime.png, shelf_animation.gif")
    print("- ridge_evolution.png, ridge_spacetime.png")
    print("- trench_evolution.png, trench_spacetime.png")


if __name__ == "__main__":
    main()
