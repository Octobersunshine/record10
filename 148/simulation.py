import numpy as np
from cosmology import Cosmology
from initial_conditions import InitialConditions
from pm_solver import PMSolver
from p3m_solver import P3MSolver

class NBodySimulation:
    def __init__(self, npart=64**3, box_size=256.0, z_init=50.0, ngrid=None, seed=42, 
                 cosmo=None, use_p3m=False, r_switch=None, r_soft=None):
        self.npart = npart
        self.box_size = box_size
        self.z_init = z_init
        self.seed = seed
        self.cosmo = cosmo if cosmo is not None else Cosmology()
        self.use_p3m = use_p3m
        
        if ngrid is None:
            ng_cbrt = int(round(npart**(1/3)))
            self.ngrid = ng_cbrt * 2
        else:
            self.ngrid = ngrid
        
        self.a_init = 1.0 / (1.0 + z_init)
        self.a = self.a_init
        self.z = z_init
        
        self.pos = None
        self.vel = None
        self.delta = None
        
        if use_p3m:
            print("Using P³M (Particle-Particle Particle-Mesh) method")
            self.solver = P3MSolver(self.ngrid, self.box_size, self.cosmo, r_switch, r_soft)
            print(f"  Switch radius: {self.solver.r_switch:.2f} Mpc/h")
            print(f"  Softening radius: {self.solver.r_soft:.2f} Mpc/h")
        else:
            print("Using PM (Particle-Mesh) method")
            self.solver = PMSolver(self.ngrid, self.box_size, self.cosmo)
        
        self.snapshots = []
        
    def initialize(self):
        print(f"Generating initial conditions at z={self.z_init:.2f}...")
        ic = InitialConditions(self.npart, self.box_size, self.z_init, self.seed, self.cosmo)
        self.pos, self.vel, self.delta = ic.generate_particles()
        self.npart = self.pos.shape[0]
        print(f"Initialized {self.npart} particles.")
        
    def drdt(self, a, vel):
        H_a = self.cosmo.H(1.0/a - 1.0)
        return vel / (a * a * H_a)
    
    def dvdt(self, a, acc):
        H_a = self.cosmo.H(1.0/a - 1.0)
        f_a = self.cosmo.f(1.0/a - 1.0)
        return -f_a * H_a * acc
    
    def lpt_kick(self, a_start, a_end, vel, acc):
        da = a_end - a_start
        a_mid = 0.5 * (a_start + a_end)
        H_mid = self.cosmo.H(1.0/a_mid - 1.0)
        f_mid = self.cosmo.f(1.0/a_mid - 1.0)
        
        dv = -f_mid * H_mid * acc * da / a_mid
        return vel + dv
    
    def drift(self, a_start, a_end, pos, vel):
        da = a_end - a_start
        a_mid = 0.5 * (a_start + a_end)
        H_mid = self.cosmo.H(1.0/a_mid - 1.0)
        
        dx = vel * da / (a_mid * a_mid * H_mid)
        pos = pos + dx
        pos = np.mod(pos, self.box_size)
        return pos
    
    def get_accelerations(self, pos, a):
        if self.use_p3m:
            acc, delta, acc_long, acc_short = self.solver.get_accelerations(pos, a)
            return acc, delta
        else:
            return self.solver.get_accelerations(pos, a)
    
    def run(self, a_final=1.0, nsteps=50):
        if self.pos is None:
            self.initialize()
        
        a_list = np.logspace(np.log10(self.a_init), np.log10(a_final), nsteps + 1)
        
        print(f"Starting simulation from a={self.a_init:.4f} (z={self.z_init:.2f}) to a={a_final:.4f} (z=0.00)")
        print(f"Using {nsteps} time steps...")
        
        acc, delta = self.get_accelerations(self.pos, self.a)
        
        for i in range(nsteps):
            a1 = a_list[i]
            a2 = a_list[i + 1]
            a_half = 0.5 * (a1 + a2)
            
            self.vel = self.lpt_kick(a1, a_half, self.vel, acc)
            
            self.pos = self.drift(a1, a2, self.pos, self.vel)
            
            acc, delta = self.get_accelerations(self.pos, a2)
            
            self.vel = self.lpt_kick(a_half, a2, self.vel, acc)
            
            self.a = a2
            self.z = 1.0 / a2 - 1.0
            self.delta = delta
            
            if (i + 1) % max(1, nsteps // 10) == 0:
                print(f"Step {i + 1}/{nsteps}: a={a2:.4f}, z={self.z:.2f}")
                self.save_snapshot()
        
        print("Simulation complete!")
        self.save_snapshot()
        
    def save_snapshot(self):
        snapshot = {
            'a': self.a,
            'z': self.z,
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            'delta': self.delta.copy()
        }
        self.snapshots.append(snapshot)
        
    def get_density_field(self):
        if self.delta is None:
            acc, delta = self.get_accelerations(self.pos, self.a)
            self.delta = delta
        return self.delta
    
    def get_final_snapshot(self):
        return self.snapshots[-1] if self.snapshots else None
