#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ionization rate calculator for atoms and molecules in strong laser fields

Models:
  - ADK (Ammosov-Delone-Krainov): Tunnel ionization (gamma << 1)
  - PPT (Perelomov-Popov-Terent'ev): Multiphoton to tunneling transition
  - MO-ADK: Molecular ADK with orientation dependence
  - Coulomb-corrected models: For HHG and LIED applications

The Keldysh parameter:
    gamma = omega * sqrt(2*Ip) / F
    - gamma << 1: Tunnel ionization
    - gamma ~ 1: Transition regime
    - gamma >> 1: Multiphoton ionization

References:
    Ammosov, Delone, Krainov, JETP 64, 1191 (1986)
    Perelomov, Popov, Terent'ev, JETP 23, 924 (1966)
    Tong, Zhao, Lin, Phys. Rev. A 66, 033402 (2002) [MO-ADK]
    Lein, Hay, Velotta, Phys. Rev. A 66, 023404 (2002) [Coulomb correction]
    Bauer, Phys. Rev. A 55, 3342 (1997)
"""

import numpy as np
from scipy import constants
from scipy.special import gamma as gamma_func, factorial, kn, iv, erf
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Callable

# Physical constants (CODATA 2018)
E_CHARGE = constants.e
HBAR = constants.hbar
M_E = constants.m_e
EPS_0 = constants.epsilon_0
C = constants.c

# Atomic units
ATOMIC_TIME = constants.value('atomic unit of time')
ATOMIC_FIELD = constants.value('atomic unit of electric field')
ATOMIC_ENERGY = constants.value('atomic unit of energy')
ATOMIC_LENGTH = constants.value('atomic unit of length')


class Atom:
    """Atom class storing electronic structure parameters"""

    ATOM_DATA = {
        'H': {'Z': 1, 'Ip': 13.6, 'l': 0, 'm': 0},
        'He': {'Z': 1.34, 'Ip': 24.59, 'l': 0, 'm': 0},
        'Ne': {'Z': 1.45, 'Ip': 21.56, 'l': 1, 'm': 0},
        'Ar': {'Z': 1.56, 'Ip': 15.76, 'l': 1, 'm': 0},
        'Kr': {'Z': 1.61, 'Ip': 14.00, 'l': 1, 'm': 0},
        'Xe': {'Z': 1.68, 'Ip': 12.13, 'l': 1, 'm': 0},
        'Cs': {'Z': 1.20, 'Ip': 3.89, 'l': 0, 'm': 0},
        'K': {'Z': 1.22, 'Ip': 4.34, 'l': 0, 'm': 0},
        'Na': {'Z': 1.25, 'Ip': 5.14, 'l': 0, 'm': 0},
    }

    def __init__(self, name: str):
        if name not in self.ATOM_DATA:
            raise ValueError(f"Unknown atom: {name}\nSupported: {list(self.ATOM_DATA.keys())}")

        data = self.ATOM_DATA[name]
        self.name = name
        self.Z = data['Z']
        self.Ip_eV = data['Ip']
        self.l = data['l']
        self.m = data['m']

        self.Ip_au = self.Ip_eV / 27.2114
        self.kappa = np.sqrt(2 * self.Ip_au)
        self.n_star = self.Z / self.kappa

    def __repr__(self):
        return (f"Atom('{self.name}', Z={self.Z}, Ip={self.Ip_eV} eV, "
                f"l={self.l}, m={self.m}, n*={self.n_star:.3f})")


class Molecule:
    """
    Molecule class for molecular ionization calculations

    Stores molecular structure and electronic properties.
    Supports diatomic molecules with orientation-dependent ionization.

    Parameters:
        name: Molecule name
        atoms: List of Atom objects
        bond_length: Bond length in atomic units
        Ip_eV: Ionization potential in eV
        orbital: Molecular orbital type ('sigma_g', 'sigma_u', 'pi_g', 'pi_u', etc.)
        symmetry: Point group symmetry
    """

    MOLECULE_DATA = {
        'H2': {'atoms': ['H', 'H'], 'bond_length': 1.40, 'Ip': 15.43,
               'orbital': 'sigma_g', 'symmetry': 'D_inf_h'},
        'N2': {'atoms': ['N', 'N'], 'bond_length': 2.08, 'Ip': 15.58,
               'orbital': 'sigma_g', 'symmetry': 'D_inf_h'},
        'O2': {'atoms': ['O', 'O'], 'bond_length': 2.28, 'Ip': 12.07,
               'orbital': 'pi_g', 'symmetry': 'D_inf_h'},
        'CO': {'atoms': ['C', 'O'], 'bond_length': 2.13, 'Ip': 14.01,
               'orbital': 'sigma', 'symmetry': 'C_inf_v'},
        'CO2': {'atoms': ['C', 'O', 'O'], 'bond_length': 2.20, 'Ip': 13.78,
                'orbital': 'pi_g', 'symmetry': 'D_inf_h'},
        'H2O': {'atoms': ['H', 'H', 'O'], 'bond_length': 1.81, 'Ip': 12.62,
                'orbital': 'b1', 'symmetry': 'C2v'},
        'CH4': {'atoms': ['C', 'H', 'H', 'H', 'H'], 'bond_length': 2.06, 'Ip': 12.61,
                'orbital': 't2', 'symmetry': 'Td'},
    }

    def __init__(self, name: str):
        if name not in self.MOLECULE_DATA:
            raise ValueError(f"Unknown molecule: {name}\nSupported: {list(self.MOLECULE_DATA.keys())}")

        data = self.MOLECULE_DATA[name]
        self.name = name
        self.bond_length = data['bond_length']
        self.Ip_eV = data['Ip']
        self.orbital = data['orbital']
        self.symmetry = data['symmetry']

        # Create atom objects
        self.atoms = []
        for atom_name in data['atoms']:
            if atom_name in Atom.ATOM_DATA:
                self.atoms.append(Atom(atom_name))
            else:
                self.atoms.append(Atom('H'))

        self.Ip_au = self.Ip_eV / 27.2114
        self.kappa = np.sqrt(2 * self.Ip_au)

        # For diatomics: effective principal quantum number
        # Using average of atoms
        if len(self.atoms) == 2:
            self.n_star = (self.atoms[0].n_star + self.atoms[1].n_star) / 2
        else:
            self.n_star = np.mean([a.n_star for a in self.atoms])

    def __repr__(self):
        return (f"Molecule('{self.name}', bond_length={self.bond_length:.2f} a.u., "
                f"Ip={self.Ip_eV} eV, orbital='{self.orbital}', symmetry='{self.symmetry}')")


class ADKIonization:
    """ADK ionization rate (tunnel ionization regime, gamma << 1)"""

    def __init__(self, atom: Atom):
        self.atom = atom

    def _c_nl(self, n_star: float, l: int) -> float:
        """Calculate C_{n^*l} coefficient"""
        numerator = 2 ** (2 * n_star)
        denominator = (n_star * gamma_func(n_star + l + 1) * gamma_func(n_star - l))
        return np.sqrt(numerator / denominator)

    def static_ionization_rate(self, F_au: float) -> float:
        """
        Static ionization rate in constant electric field

        Parameters:
            F_au: Electric field strength (atomic units)

        Returns:
            Ionization rate Gamma (atomic units)
        """
        if F_au <= 0:
            return 0.0

        atom = self.atom
        n_star = atom.n_star
        l = atom.l
        m = atom.m
        kappa = atom.kappa

        exponent = 2 * n_star - abs(m) - 1
        C_nl = self._c_nl(n_star, l)

        rate = (
            (2 * kappa**3 / F_au) ** exponent
            * C_nl**2
            * np.exp(-2 * kappa**3 / (3 * F_au))
        )

        return rate

    def cycle_averaged_rate(self, F0_au: float, omega_au: float) -> float:
        """
        Cycle-averaged ionization rate

        Parameters:
            F0_au: Electric field amplitude (atomic units)
            omega_au: Laser frequency (atomic units)

        Returns:
            Cycle-averaged ionization rate (atomic units)
        """
        def integrand(phase):
            F = F0_au * np.cos(phase)
            if abs(F) < 1e-30:
                return 0.0
            return self.static_ionization_rate(abs(F))

        result, _ = quad(integrand, 0, 2 * np.pi)
        return result / (2 * np.pi)

    def calculate_survival(self, F0_au: float, omega_au: float,
                           n_cycles: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate survival probability vs time

        Parameters:
            F0_au: Electric field amplitude (atomic units)
            omega_au: Laser frequency (atomic units)
            n_cycles: Number of optical cycles

        Returns:
            (time array, survival probability array)
        """
        cycle = 2 * np.pi / omega_au
        t_total = n_cycles * cycle
        n_points = 1000
        t = np.linspace(0, t_total, n_points)

        dt = t[1] - t[0]
        survival = np.ones(n_points)

        for i in range(1, n_points):
            phase = omega_au * t[i]
            F = abs(F0_au * np.cos(phase))
            rate = self.static_ionization_rate(F)
            survival[i] = survival[i - 1] * np.exp(-rate * dt)

        return t, survival


class PPTIonization:
    """
    PPT (Perelomov-Popov-Terent'ev) ionization rate

    Unified description from multiphoton to tunneling regime.
    """

    def __init__(self, atom: Atom):
        self.atom = atom

    def _c_nl(self, n_star: float, l: int) -> float:
        """Calculate C_{n^*l} coefficient"""
        numerator = 2 ** (2 * n_star)
        denominator = (n_star * gamma_func(n_star + l + 1) * gamma_func(n_star - l))
        return np.sqrt(numerator / denominator)

    def keldysh_parameter(self, F_au: float, omega_au: float) -> float:
        """Calculate Keldysh parameter gamma = omega * sqrt(2*Ip) / F"""
        if F_au <= 0:
            return np.inf
        return omega_au * self.atom.kappa / F_au

    def g_function(self, gamma: float) -> float:
        """
        PPT g(gamma) function
        g(gamma) = 3/(2*gamma) * [arcsinh(gamma) - gamma/sqrt(1+gamma^2)]
        """
        if gamma == 0:
            return 1.0
        if gamma < 1e-6:
            return 1.0 - 2.0 * gamma**2 / 5.0
        if gamma > 1e10:
            return 1.0 / gamma**2

        arcsinh_gamma = np.log(gamma + np.sqrt(gamma**2 + 1))
        return 3.0 / (2.0 * gamma) * (arcsinh_gamma - gamma / np.sqrt(1.0 + gamma**2))

    def alpha_function(self, gamma: float) -> float:
        """PPT alpha(gamma) = 2 * (arcsinh(gamma) - gamma/sqrt(1+gamma^2))"""
        if gamma == 0:
            return 0.0
        arcsinh_gamma = np.log(gamma + np.sqrt(gamma**2 + 1))
        return 2.0 * (arcsinh_gamma - gamma / np.sqrt(1.0 + gamma**2))

    def beta_function(self, gamma: float) -> float:
        """PPT beta(gamma) = 2*gamma / sqrt(1+gamma^2)"""
        if gamma == np.inf:
            return 2.0
        return 2.0 * gamma / np.sqrt(1.0 + gamma**2)

    def _phi_sum(self, gamma: float, nu: float, n_star: float, m: int,
                 max_terms: int = 200) -> float:
        """
        PPT summation Phi(gamma, nu)
        Phi = sum_n exp(-n*alpha) * I_{|m|}(n*beta)
        """
        if gamma == 0:
            return 1.0

        alpha = self.alpha_function(gamma)
        beta = self.beta_function(gamma)
        n_min = int(np.ceil(nu))

        total = 0.0
        exp_neg_alpha = np.exp(-alpha)
        m_abs = abs(m)

        for n in range(n_min, n_min + max_terms):
            term = exp_neg_alpha**n * iv(m_abs, n * beta)
            total += term
            if abs(term) < 1e-30 * abs(total):
                break

        return total

    def ionization_rate(self, F_au: float, omega_au: float) -> float:
        """
        PPT cycle-averaged ionization rate (linear polarization)

        Parameters:
            F_au: Electric field amplitude (atomic units)
            omega_au: Laser frequency (atomic units)

        Returns:
            Cycle-averaged ionization rate (atomic units)
        """
        if F_au <= 0:
            return 0.0

        atom = self.atom
        n_star = atom.n_star
        l = atom.l
        m = atom.m
        kappa = atom.kappa
        Ip_au = atom.Ip_au

        gamma = self.keldysh_parameter(F_au, omega_au)
        nu = Ip_au / omega_au
        C_nl = self._c_nl(n_star, l)
        g = self.g_function(gamma)

        exp_arg = -2.0 * kappa**3 / (3.0 * F_au) * g

        prefactor = (
            4.0 * omega_au / np.sqrt(3.0 * np.pi)
            * C_nl**2 / gamma_func(n_star - l)
            * (2.0 * kappa**3 / F_au) ** (2.0 * n_star - abs(m) - 0.5)
        )

        phi = self._phi_sum(gamma, nu, n_star, m)

        return prefactor * np.exp(exp_arg) * phi

    def static_ionization_rate(self, F_au: float) -> float:
        """Static ionization rate (ADK limit for gamma -> 0)"""
        adk = ADKIonization(self.atom)
        return adk.static_ionization_rate(F_au)

    def calculate_survival(self, F0_au: float, omega_au: float,
                           n_cycles: int = 10,
                           use_adk_static: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate survival probability vs time"""
        cycle = 2 * np.pi / omega_au
        t_total = n_cycles * cycle
        n_points = 1000
        t = np.linspace(0, t_total, n_points)

        dt = t[1] - t[0]
        survival = np.ones(n_points)

        if use_adk_static:
            for i in range(1, n_points):
                phase = omega_au * t[i]
                F = abs(F0_au * np.cos(phase))
                rate = self.static_ionization_rate(F)
                survival[i] = survival[i - 1] * np.exp(-rate * dt)
        else:
            avg_rate = self.ionization_rate(F0_au, omega_au)
            survival = np.exp(-avg_rate * t)

        return t, survival


class MOADKIonization:
    """
    MO-ADK (Molecular ADK) with orientation-dependent ionization

    Extends ADK to diatomic molecules with orientation-dependent ionization rates.
    Based on Tong, Zhao, Lin, Phys. Rev. A 66, 033402 (2002).

    The molecular ionization rate depends on:
    - Angle theta between molecular axis and laser polarization
    - Molecular orbital symmetry (sigma_g, sigma_u, pi_g, pi_u, etc.)
    - Electronic structure factors
    """

    def __init__(self, molecule: Molecule):
        self.molecule = molecule

    def _spherical_harmonic_factor(self, theta: float, l: int, m: int) -> float:
        """
        Calculate angular factor from spherical harmonics

        For HOMO of diatomics, typically:
        sigma orbitals: L=0 (s-like) or L=2 (d-like) contributions
        pi orbitals: L=1 (p-like) contributions
        """
        if l == 0:
            return 1.0 / np.sqrt(4 * np.pi)
        elif l == 1:
            if m == 0:
                return np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
            elif abs(m) == 1:
                return np.sqrt(3 / (8 * np.pi)) * np.sin(theta)
        elif l == 2:
            if m == 0:
                return np.sqrt(5 / (16 * np.pi)) * (3 * np.cos(theta)**2 - 1)
            elif abs(m) == 1:
                return np.sqrt(15 / (8 * np.pi)) * np.sin(theta) * np.cos(theta)
            elif abs(m) == 2:
                return np.sqrt(15 / (32 * np.pi)) * np.sin(theta)**2
        return 1.0

    def _orbital_angular_factor(self, theta: float, orbital_type: str) -> float:
        """
        Calculate angular factor for different molecular orbital types

        Based on the MO-ADK model:
        - sigma_g: bonding sigma orbital (HOMO of H2, N2)
        - sigma_u: antibonding sigma orbital
        - pi_g: bonding pi orbital (HOMO of O2)
        - pi_u: antibonding pi orbital
        """
        if orbital_type.startswith('sigma'):
            # sigma orbitals: cos^n(theta) dependence
            # For sigma_g (H2, N2): max at theta = 0 (along axis)
            if 'g' in orbital_type:
                # Bonding sigma: primarily s + pz + dz2 mixing
                # Typical form: a*cos^2(theta) + b
                return 0.8 * np.cos(theta)**2 + 0.2
            else:
                # sigma_u: pz-like, max at theta = 0
                return np.cos(theta)**2

        elif orbital_type.startswith('pi'):
            # pi orbitals: sin^2(theta) dependence
            # Max at theta = 90 degrees (perpendicular to axis)
            if 'g' in orbital_type:
                # pi_g: bonding pi (O2 HOMO)
                return np.sin(theta)**2
            else:
                # pi_u: antibonding pi
                return np.sin(theta)**2

        elif orbital_type == 'b1':
            # H2O HOMO: p_y orbital
            return np.sin(theta)**2

        elif orbital_type == 't2':
            # CH4 HOMO: p-like, roughly isotropic with some modulation
            return 1.0 + 0.3 * np.cos(theta)**2

        else:
            # Default: isotropic
            return 1.0

    def _interference_factor(self, theta: float, R: float, kappa: float) -> float:
        """
        Calculate two-center interference factor for diatomics

        For diatomic molecules with internuclear distance R:
        The electron wavefunction from two centers interferes.

        Parameters:
            theta: angle between molecular axis and polarization
            R: internuclear distance (a.u.)
            kappa: asymptotic decay constant (a.u.)
        """
        if R == 0:
            return 1.0

        # Phase difference between two centers
        # This is a simplified model; full MO-ADK uses actual molecular wavefunctions
        phase = kappa * R * np.cos(theta)

        # Two-center interference
        return 2.0 * (1.0 + np.exp(-kappa * R) * np.cos(phase))

    def ionization_rate(self, F_au: float, theta: float) -> float:
        """
        MO-ADK ionization rate with orientation dependence

        Parameters:
            F_au: Electric field strength (atomic units)
            theta: Angle between molecular axis and laser polarization (radians)

        Returns:
            Ionization rate Gamma (atomic units)
        """
        if F_au <= 0:
            return 0.0

        molecule = self.molecule
        kappa = molecule.kappa
        n_star = molecule.n_star

        # Get angular factor from orbital symmetry
        angular_factor = self._orbital_angular_factor(theta, molecule.orbital)

        # Two-center interference factor (for diatomics)
        if len(molecule.atoms) == 2:
            interference = self._interference_factor(theta, molecule.bond_length, kappa)
        else:
            interference = 1.0

        # Basic ADK-like rate
        # Using effective principal quantum number
        C_factor = 2 ** (2 * n_star) / (n_star * gamma_func(n_star + 1)**2)

        rate = (
            angular_factor
            * interference
            * C_factor
            * (2 * kappa**3 / F_au) ** (2 * n_star - 1)
            * np.exp(-2 * kappa**3 / (3 * F_au))
        )

        return max(rate, 0.0)

    def orientation_averaged_rate(self, F_au: float) -> float:
        """
        Calculate orientation-averaged ionization rate

        Average over all molecular orientations:
        <Gamma> = (1/4*pi) * int Gamma(theta) * sin(theta) dtheta dphi

        Parameters:
            F_au: Electric field strength (atomic units)

        Returns:
            Orientation-averaged ionization rate (atomic units)
        """
        def integrand(theta):
            return self.ionization_rate(F_au, theta) * np.sin(theta)

        result, _ = quad(integrand, 0, np.pi)
        return result / 2.0  # average over 2*pi from phi

    def angular_distribution(self, F_au: float, n_angles: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ionization rate angular distribution

        Parameters:
            F_au: Electric field strength (atomic units)
            n_angles: Number of angle points

        Returns:
            (theta array, rate array)
        """
        thetas = np.linspace(0, np.pi, n_angles)
        rates = np.array([self.ionization_rate(F_au, theta) for theta in thetas])
        return thetas, rates


class CoulombCorrectedIonization:
    """
    Coulomb-corrected ionization models for HHG and LIED applications

    The Coulomb potential of the parent ion significantly affects:
    1. Tunneling ionization rate (Coulomb suppression/enhancement)
    2. Electron trajectory after ionization (important for HHG and LIED)
    3. Recollision dynamics

    References:
        Lein et al., Phys. Rev. A 66, 023404 (2002)
        Bauer, Phys. Rev. A 55, 3342 (1997)
    """

    def __init__(self, atom: Atom, charge: int = 1):
        """
        Parameters:
            atom: Atom object
            charge: Charge of the parent ion after ionization (default: +1)
        """
        self.atom = atom
        self.charge = charge
        self.adk = ADKIonization(atom)

    def coulomb_correction_factor(self, F_au: float) -> float:
        """
        Calculate Coulomb correction factor for ionization rate

        The Coulomb potential lowers the effective barrier, enhancing ionization.
        This factor multiplies the ADK rate.

        Based on the model by Bauer (1997):
        C(F) = exp( sqrt(2*Ip)/F * (Z - 1) * [1 - 1/sqrt(1 + (2*kappa/F)^2)] )

        where Z is the nuclear charge and Ip is the ionization potential.
        """
        if F_au <= 0:
            return 1.0

        kappa = self.atom.kappa
        Z = self.atom.Z

        # The Coulomb correction is most significant for lower fields
        # and for higher ion charges
        if self.charge > 1:
            Z_eff = self.charge
        else:
            Z_eff = Z - 1  # for single ionization

        # Simple Coulomb correction factor
        # Enhanced tunneling due to Coulomb attraction
        arg = (Z_eff * kappa) / (F_au + 1e-10)
        correction = np.exp(np.sqrt(2) * arg * (1.0 - 1.0 / np.sqrt(1.0 + (2 * kappa / F_au)**2)))

        return correction

    def corrected_ionization_rate(self, F_au: float) -> float:
        """
        Coulomb-corrected static ionization rate

        Parameters:
            F_au: Electric field strength (atomic units)

        Returns:
            Corrected ionization rate (atomic units)
        """
        adk_rate = self.adk.static_ionization_rate(F_au)
        correction = self.coulomb_correction_factor(F_au)
        return adk_rate * correction

    def tunnel_exit_position(self, F_au: float) -> float:
        """
        Calculate electron exit position after tunneling

        In the strong-field approximation, this is the position where
        the electron exits the tunneling barrier.

        Parameters:
            F_au: Electric field strength (atomic units)

        Returns:
            Exit position (atomic units)
        """
        if F_au <= 0:
            return np.inf

        kappa = self.atom.kappa

        # Simple model: exit at the point where V(x) = -Ip
        # V(x) = -Z/x - F*x = -Ip
        # Solve for x
        Z = self.atom.Z

        def V(x):
            return -Z / x - F_au * x + self.atom.Ip_au

        try:
            # Find root between 0.1 and 20 a.u.
            x_exit = brentq(V, 0.1, 50.0)
        except ValueError:
            # If no root found, use approximate formula
            x_exit = np.sqrt(2 * Z / F_au)

        return x_exit

    def initial_momentum_distribution(self, F_au: float, p: float) -> float:
        """
        Calculate initial transverse momentum distribution after tunneling

        The distribution is a Gaussian:
        f(p_perp) = exp(-p_perp^2 / sigma^2)

        where sigma^2 = F / (2*kappa) for ADK.

        Parameters:
            F_au: Electric field strength (atomic units)
            p: Transverse momentum (atomic units)

        Returns:
            Probability density
        """
        if F_au <= 0:
            return 0.0

        kappa = self.atom.kappa
        sigma2 = F_au / (2 * kappa)

        return np.exp(-p**2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

    def coulomb_trajectory_correction(self, t: float, v0: float, F_au: float) -> Tuple[float, float]:
        """
        Calculate Coulomb-corrected electron trajectory

        Solves the classical equation of motion:
        d^2x/dt^2 = F(t) - Z / x^2

        Parameters:
            t: Time since ionization (atomic units)
            v0: Initial velocity (atomic units)
            F_au: Electric field strength (atomic units)

        Returns:
            (position, velocity) at time t
        """
        Z = self.atom.Z
        x = self.tunnel_exit_position(F_au)
        v = v0

        dt = 0.01  # time step
        n_steps = int(t / dt)

        for _ in range(n_steps):
            # Simple Verlet integration
            a = F_au - Z / max(x, 0.01)**2
            v += a * dt
            x += v * dt

            if x < 0:
                # Collision with nucleus
                x = 0.0
                v = 0.0
                break

        return x, v

    def hhg_cutoff_energy(self, F_au: float, omega_au: float) -> float:
        """
        Calculate HHG cutoff energy with Coulomb corrections

        The standard cutoff law is: E_cutoff = Ip + 3.17*Up
        where Up = F^2/(4*omega^2) is the ponderomotive energy.

        Coulomb corrections slightly modify this.

        Parameters:
            F_au: Electric field amplitude (atomic units)
            omega_au: Laser frequency (atomic units)

        Returns:
            Cutoff energy (atomic units)
        """
        Up = F_au**2 / (4 * omega_au**2)
        Ip = self.atom.Ip_au

        # Standard cutoff
        cutoff_standard = Ip + 3.17 * Up

        # Coulomb correction factor (typically small, ~5-10%)
        correction = 1.0 + 0.05 * np.log10(Up / Ip + 1)

        return cutoff_standard * correction

    def lied_differential_cross_section(self, F_au: float, omega_au: float,
                                        scattering_angle: float) -> float:
        """
        Calculate LIED (Laser-Induced Electron Diffraction) differential cross section

        This is a simplified model based on recollision of the tunnel-ionized
        electron with the parent ion.

        Parameters:
            F_au: Electric field amplitude (atomic units)
            omega_au: Laser frequency (atomic units)
            scattering_angle: Scattering angle (radians)

        Returns:
            Differential cross section (arbitrary units)
        """
        Up = F_au**2 / (4 * omega_au**2)
        E_collision = 3.17 * Up  # Typical recollision energy

        # Simple Rutherford-like scattering
        # dsigma/dOmega ~ 1/sin^4(theta/2) * exp(-b*sin^2(theta/2))
        # where b accounts for the finite extent of the ion

        sin_half_theta = np.sin(scattering_angle / 2.0)
        if sin_half_theta < 1e-10:
            return 0.0

        # Screening parameter
        kappa = self.atom.kappa
        b = 2 * kappa * np.sqrt(2 * E_collision)

        # Modified Rutherford formula
        dsigma = (1.0 / sin_half_theta**4) * np.exp(-b * sin_half_theta**2)

        return dsigma


def intensity_to_field(intensity_wcm2: float) -> float:
    """Convert intensity (W/cm2) to electric field amplitude (atomic units)"""
    intensity_si = intensity_wcm2 * 1e4
    E_vm = np.sqrt(2 * intensity_si / (C * EPS_0))
    return E_vm / ATOMIC_FIELD


def field_to_intensity(F_au: float) -> float:
    """Convert electric field (atomic units) to intensity (W/cm2)"""
    E_vm = F_au * ATOMIC_FIELD
    intensity_si = 0.5 * C * EPS_0 * E_vm**2
    return intensity_si / 1e4


def wavelength_to_omega(wavelength_nm: float) -> float:
    """Convert laser wavelength (nm) to frequency (atomic units)"""
    wavelength_m = wavelength_nm * 1e-9
    omega_si = 2 * np.pi * C / wavelength_m
    return omega_si * ATOMIC_TIME


def main():
    """Main function: Demonstrate all ionization models"""

    print("=" * 70)
    print("  Strong Field Ionization Calculator")
    print("  (ADK, PPT, MO-ADK, Coulomb-corrected models)")
    print("=" * 70)

    # Laser parameters
    wavelength_nm = 800
    omega_au = wavelength_to_omega(wavelength_nm)

    print(f"\nLaser parameters:")
    print(f"  Wavelength: {wavelength_nm} nm")
    print(f"  Frequency (a.u.): {omega_au:.6f}")
    print(f"  Photon energy: {omega_au * 27.2114:.4f} eV")
    print(f"  Period (fs): {2*np.pi/omega_au * ATOMIC_TIME * 1e15:.2f}")

    # =========================================================================
    # Figure 1: ADK vs PPT comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("  1. ADK vs PPT Ionization Rates")
    print("=" * 70)

    atoms_to_calculate = ['H', 'Ar', 'Xe']
    intensities = np.logspace(11, 16, 100)

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    for idx, atom_name in enumerate(atoms_to_calculate):
        ax = axes1[idx]

        atom = Atom(atom_name)
        adk = ADKIonization(atom)
        ppt = PPTIonization(atom)

        rates_adk = []
        rates_ppt = []
        gammas = []

        for I in intensities:
            F0 = intensity_to_field(I)
            rate_adk = adk.cycle_averaged_rate(F0, omega_au)
            rate_ppt = ppt.ionization_rate(F0, omega_au)
            gamma = ppt.keldysh_parameter(F0, omega_au)

            rates_adk.append(rate_adk)
            rates_ppt.append(rate_ppt)
            gammas.append(gamma)

        rates_adk = np.array(rates_adk) / ATOMIC_TIME
        rates_ppt = np.array(rates_ppt) / ATOMIC_TIME
        gammas = np.array(gammas)

        ax.loglog(intensities, rates_ppt, 'r-', label='PPT', linewidth=2.5)
        ax.loglog(intensities, rates_adk, 'b--', label='ADK', linewidth=1.5, alpha=0.8)

        gamma_one_idx = np.argmin(np.abs(gammas - 1.0))
        I_gamma_one = intensities[gamma_one_idx]
        ax.axvline(x=I_gamma_one, color='k', linestyle=':', alpha=0.5, label='gamma=1')

        ax.set_xlabel('Intensity (W/cm2)', fontsize=12)
        ax.set_ylabel('Ionization rate (1/s)', fontsize=12)
        ax.set_title(f'{atom.name} atom (Ip = {atom.Ip_eV} eV)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1e11, 1e16)

        print(f"\n{atom_name} atom:")
        print(f"  Transition intensity (gamma=1): {I_gamma_one:.2e} W/cm2")
        for I_ref in [1e13, 1e14, 1e15]:
            F0_ref = intensity_to_field(I_ref)
            rate_ppt_ref = ppt.ionization_rate(F0_ref, omega_au) / ATOMIC_TIME
            gamma_ref = ppt.keldysh_parameter(F0_ref, omega_au)
            regime = "multiphoton" if gamma_ref > 1.5 else ("transition" if gamma_ref > 0.3 else "tunneling")
            print(f"    I = {I_ref:.0e} W/cm2: Gamma = {rate_ppt_ref:.2e} s^-1, gamma = {gamma_ref:.2f} ({regime})")

    fig1.suptitle('ADK vs PPT Ionization Rates (800 nm)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('adk_vs_ppt.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: adk_vs_ppt.png")

    # =========================================================================
    # Figure 2: MO-ADK orientation dependence
    # =========================================================================
    print("\n" + "=" * 70)
    print("  2. MO-ADK: Molecular Orientation Dependence")
    print("=" * 70)

    molecules = ['H2', 'N2', 'O2', 'CO2']
    intensity = 1e14
    F0 = intensity_to_field(intensity)

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    ax_polar = plt.subplot(231, projection='polar')

    for idx, mol_name in enumerate(molecules):
        ax = axes2[idx // 2, idx % 2]

        mol = Molecule(mol_name)
        mo_adk = MOADKIonization(mol)

        thetas, rates = mo_adk.angular_distribution(F0)
        rates_norm = rates / np.max(rates)

        ax.plot(thetas * 180 / np.pi, rates_norm, 'b-', linewidth=2)
        ax.set_xlabel('Angle theta (degrees)', fontsize=12)
        ax.set_ylabel('Normalized rate', fontsize=12)
        ax.set_title(f'{mol.name} ({mol.orbital})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.05)

        if idx == 0:
            ax_polar.plot(thetas, rates_norm, 'b-', linewidth=2)
            ax_polar.set_title('H2 angular distribution', fontsize=12)

        avg_rate = mo_adk.orientation_averaged_rate(F0) / ATOMIC_TIME
        print(f"\n{mol_name} molecule:")
        print(f"  Orbital: {mol.orbital}, Bond length: {mol.bond_length:.2f} a.u.")
        print(f"  Max rate (theta=0): {np.max(rates) / ATOMIC_TIME:.2e} s^-1")
        print(f"  Orientation-averaged rate: {avg_rate:.2e} s^-1")

    fig2.suptitle(f'Molecular Ionization Angular Dependence (I = {intensity:.0e} W/cm2)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('moadk_orientation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: moadk_orientation.png")

    # =========================================================================
    # Figure 3: Coulomb corrections
    # =========================================================================
    print("\n" + "=" * 70)
    print("  3. Coulomb Corrections for HHG and LIED")
    print("=" * 70)

    atom = Atom('Ar')
    coulomb = CoulombCorrectedIonization(atom)
    adk = ADKIonization(atom)

    F_range = np.logspace(-3, -0.5, 50)
    I_range = [field_to_intensity(F) for F in F_range]

    correction_factors = [coulomb.coulomb_correction_factor(F) for F in F_range]
    rates_adk = [adk.static_ionization_rate(F) for F in F_range]
    rates_coulomb = [coulomb.corrected_ionization_rate(F) for F in F_range]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    ax3a.semilogx(I_range, correction_factors, 'r-', linewidth=2)
    ax3a.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No correction')
    ax3a.set_xlabel('Intensity (W/cm2)', fontsize=12)
    ax3a.set_ylabel('Coulomb correction factor', fontsize=12)
    ax3a.set_title('Coulomb Enhancement Factor', fontsize=14)
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)

    ax3b.loglog(I_range, rates_adk, 'b--', label='ADK', linewidth=1.5)
    ax3b.loglog(I_range, rates_coulomb, 'r-', label='Coulomb-corrected', linewidth=2)
    ax3b.set_xlabel('Intensity (W/cm2)', fontsize=12)
    ax3b.set_ylabel('Ionization rate (a.u.)', fontsize=12)
    ax3b.set_title('Ionization Rate Comparison', fontsize=14)
    ax3b.legend(fontsize=10)
    ax3b.grid(True, alpha=0.3)

    # Tunnel exit position
    fig3b, ax3c = plt.subplots(figsize=(10, 5))
    exit_positions = [coulomb.tunnel_exit_position(F) for F in F_range]
    ax3c.semilogx(I_range, exit_positions, 'b-', linewidth=2)
    ax3c.set_xlabel('Intensity (W/cm2)', fontsize=12)
    ax3c.set_ylabel('Tunnel exit position (a.u.)', fontsize=12)
    ax3c.set_title('Electron Exit Position After Tunneling (Ar)', fontsize=14)
    ax3c.grid(True, alpha=0.3)

    print("\nArgon Coulomb effects:")
    for I_ref in [1e13, 1e14, 1e15]:
        F_ref = intensity_to_field(I_ref)
        corr = coulomb.coulomb_correction_factor(F_ref)
        x_exit = coulomb.tunnel_exit_position(F_ref)
        print(f"  I = {I_ref:.0e} W/cm2:")
        print(f"    Correction factor: {corr:.2f}")
        print(f"    Exit position: {x_exit:.2f} a.u. = {x_exit * ATOMIC_LENGTH * 1e10:.2f} A")

    plt.tight_layout()
    plt.savefig('coulomb_corrections.png', dpi=150, bbox_inches='tight')
    plt.savefig('exit_position.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved as: coulomb_corrections.png, exit_position.png")

    # =========================================================================
    # Figure 4: HHG and LIED calculations
    # =========================================================================
    print("\n" + "=" * 70)
    print("  4. HHG Cutoff and LIED Cross Sections")
    print("=" * 70)

    intensities_hhg = np.logspace(13, 15, 30)
    cutoffs = []

    for I in intensities_hhg:
        F = intensity_to_field(I)
        cutoff = coulomb.hhg_cutoff_energy(F, omega_au)
        cutoffs.append(cutoff)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    ax4a.semilogx(intensities_hhg, np.array(cutoffs) * 27.2114, 'b-', linewidth=2)
    ax4a.set_xlabel('Intensity (W/cm2)', fontsize=12)
    ax4a.set_ylabel('HHG cutoff energy (eV)', fontsize=12)
    ax4a.set_title('HHG Cutoff vs Intensity (Ar, 800 nm)', fontsize=14)
    ax4a.grid(True, alpha=0.3)

    # LIED differential cross section
    F_lied = intensity_to_field(5e14)
    angles = np.linspace(0.1, np.pi - 0.1, 100)
    dsigma = [coulomb.lied_differential_cross_section(F_lied, omega_au, theta) for theta in angles]

    ax4b.semilogy(angles * 180 / np.pi, dsigma, 'r-', linewidth=2)
    ax4b.set_xlabel('Scattering angle (degrees)', fontsize=12)
    ax4b.set_ylabel('d sigma / d Omega (a.u.)', fontsize=12)
    ax4b.set_title('LIED Differential Cross Section (Ar, 5e14 W/cm2)', fontsize=14)
    ax4b.grid(True, alpha=0.3)

    print("\nHHG Cutoff energies (Ar, 800 nm):")
    for I_ref in [1e14, 5e14, 1e15]:
        F_ref = intensity_to_field(I_ref)
        Up = F_ref**2 / (4 * omega_au**2) * 27.2114
        cutoff_eV = coulomb.hhg_cutoff_energy(F_ref, omega_au) * 27.2114
        print(f"  I = {I_ref:.0e} W/cm2: Up = {Up:.1f} eV, Cutoff = {cutoff_eV:.1f} eV")

    plt.tight_layout()
    plt.savefig('hhg_lied.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: hhg_lied.png")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Summary: Ionization Parameters")
    print("=" * 70)

    print(f"\nAtoms at I = 1e14 W/cm2:")
    print(f"{'Atom':<6} {'Ip (eV)':<10} {'Gamma_ADK':<14} {'Gamma_PPT':<14} {'Gamma_Coulomb':<14}")
    print("-" * 60)

    F_ref = intensity_to_field(1e14)
    for atom_name in ['H', 'He', 'Ar', 'Xe']:
        atom = Atom(atom_name)
        adk = ADKIonization(atom)
        ppt = PPTIonization(atom)
        coulomb = CoulombCorrectedIonization(atom)

        rate_adk = adk.cycle_averaged_rate(F_ref, omega_au) / ATOMIC_TIME
        rate_ppt = ppt.ionization_rate(F_ref, omega_au) / ATOMIC_TIME
        rate_coulomb = coulomb.corrected_ionization_rate(F_ref) / ATOMIC_TIME

        print(f"{atom_name:<6} {atom.Ip_eV:<10.2f} {rate_adk:<14.2e} "
              f"{rate_ppt:<14.2e} {rate_coulomb:<14.2e}")

    print("\nMolecules at I = 1e14 W/cm2:")
    print(f"{'Molecule':<10} {'Ip (eV)':<10} {'Orbital':<10} {'Gamma_max':<14} {'Gamma_avg':<14}")
    print("-" * 60)

    for mol_name in ['H2', 'N2', 'O2', 'CO2']:
        mol = Molecule(mol_name)
        mo_adk = MOADKIonization(mol)

        _, rates = mo_adk.angular_distribution(F_ref)
        rate_max = np.max(rates) / ATOMIC_TIME
        rate_avg = mo_adk.orientation_averaged_rate(F_ref) / ATOMIC_TIME

        print(f"{mol_name:<10} {mol.Ip_eV:<10.2f} {mol.orbital:<10} "
              f"{rate_max:<14.2e} {rate_avg:<14.2e}")


if __name__ == '__main__':
    main()
