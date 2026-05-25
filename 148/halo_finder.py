import numpy as np
from scipy.spatial import cKDTree

class FriendsOfFriends:
    def __init__(self, pos, box_size, link_length=0.2, n_min=20):
        self.pos = pos
        self.box_size = box_size
        self.link_length = link_length
        self.n_min = n_min
        self.npart = pos.shape[0]
        
        self.halos = []
        self.group_ids = -np.ones(self.npart, dtype=np.int32)
        
    def find_halos(self):
        print(f"Running FoF halo finder with link_length={self.link_length} Mpc/h...")
        
        tree = cKDTree(self.pos, boxsize=self.box_size)
        
        group_id = 0
        for i in range(self.npart):
            if self.group_ids[i] != -1:
                continue
            
            neighbors = tree.query_ball_point(self.pos[i], self.link_length)
            
            if len(neighbors) < self.n_min:
                continue
            
            group_members = []
            to_check = set(neighbors)
            
            while to_check:
                idx = to_check.pop()
                if self.group_ids[idx] == -1:
                    self.group_ids[idx] = group_id
                    group_members.append(idx)
                    new_neighbors = tree.query_ball_point(self.pos[idx], self.link_length)
                    for n in new_neighbors:
                        if self.group_ids[n] == -1:
                            to_check.add(n)
            
            if len(group_members) >= self.n_min:
                self.halos.append({
                    'id': group_id,
                    'members': np.array(group_members),
                    'size': len(group_members)
                })
                group_id += 1
        
        print(f"Found {len(self.halos)} halos with >= {self.n_min} particles")
        return self.halos
    
    def calculate_halo_properties(self, particle_mass=None):
        if particle_mass is None:
            particle_mass = 1.0
        
        for halo in self.halos:
            members = halo['members']
            halo_pos = self.pos[members]
            
            com = np.mean(halo_pos, axis=0)
            dists = np.linalg.norm(halo_pos - com, axis=1)
            r_max = np.max(dists)
            
            halo['center'] = com
            halo['radius'] = r_max
            halo['mass'] = len(members) * particle_mass
        
        return self.halos

def mass_function(halos, box_size, n_bins=10):
    masses = np.array([h['mass'] for h in halos])
    
    if len(masses) == 0:
        return np.array([]), np.array([]), np.array([])
    
    mass_bins = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), n_bins + 1)
    
    n_halos, _ = np.histogram(masses, bins=mass_bins)
    
    dlogM = np.log10(mass_bins[1:]) - np.log10(mass_bins[:-1])
    volume = box_size**3
    dndlogM = n_halos / (volume * dlogM)
    
    mass_mid = np.sqrt(mass_bins[1:] * mass_bins[:-1])
    
    return mass_mid, dndlogM, mass_bins

def plot_mass_function(mass_mid, dndlogM, z=0.0, output_file='mass_function.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    mask = dndlogM > 0
    ax.loglog(mass_mid[mask], dndlogM[mask], 'o-')
    
    ax.set_xlabel('Mass [M$_\odot$/h]')
    ax.set_ylabel('dn/dlogM [h$^3$ Mpc$^{-3}$]')
    ax.set_title(f'Halo Mass Function (z={z:.2f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved mass function to {output_file}")

def find_halos_and_mass_function(pos, box_size, link_length=0.2, n_min=20, particle_mass=None):
    fof = FriendsOfFriends(pos, box_size, link_length, n_min)
    halos = fof.find_halos()
    halos = fof.calculate_halo_properties(particle_mass)
    
    mass_mid, dndlogM, mass_bins = mass_function(halos, box_size)
    
    return halos, mass_mid, dndlogM
