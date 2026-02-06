"""
Vector Calculus Simulation for Dimensional Collapse
Uses FiPy for PDE (Information Flow Field) and scipy for ODE (Differential Collapse)

Implements:
1. Information Flow Field: F(n,ρ,D) = ∇Ψ with divergence analysis
2. Differential Collapse Equation: dΨ/dn with phase transition detection

Date: February 5, 2026
"""

import numpy as np
import csv
import os
from datetime import datetime
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import FiPy, fall back to manual implementation if not available
try:
    from fipy import Grid2D, CellVariable, DiffusionTerm, TransientTerm, Viewer
    FIPY_AVAILABLE = True
except ImportError:
    print("Warning: FiPy not available. Using manual PDE implementation.")
    FIPY_AVAILABLE = False

# ==================== MODEL PARAMETERS ====================

PARAMS_3D = {
    'a': -4.506,
    'b': 20.809,
    'lambda': 0.896,
    'n0': 4.432
}

PARAMS_4D = {
    'a': -10.899,
    'b': 45.211,
    'lambda': 0.292,
    'n0': -0.172
}

COVERAGE_PARAMS = {
    'a_max': 0.317,
    'b_max': 0.261,
    'a_peak': 16.093,
    'b_peak': -7.043,
    'sigma': 4.352,
    'beta': -0.001492
}

COLLAPSE_PARAMS = {
    'alpha': 0.1,
    'beta': 0.5,
    'sigma_max': 0.5
}

# ==================== HELPER FUNCTIONS ====================

def load_csv_data(filepath: str) -> List[Dict]:
    """Load CSV data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clusters_key = 'avg_3d_clusters' if 'avg_3d_clusters' in row else 'avg_4d_clusters'
            data.append({
                'n': int(row['num_vars']),
                'density': float(row['density']),
                'clusters': float(row.get(clusters_key, 0)),
                'info_density': float(row.get('avg_information_density', 0)),
                'coverage': float(row.get('avg_coverage_ratio', 0)),
                'uniqueness': float(row.get('avg_uniqueness_ratio', 0))
            })
    return data


def i_avg_model(n: float, rho: float, dimension: int) -> float:
    """Information saturation model"""
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    i_sat = params['a'] + params['b'] * rho
    exponent = -params['lambda'] * (n - params['n0'])
    return i_sat * (1 - np.exp(exponent))


def gamma_model(n: float, rho: float) -> float:
    """Coverage constriction model (3D)"""
    p = COVERAGE_PARAMS
    gamma_max = p['a_max'] + p['b_max'] * rho
    n_peak = p['a_peak'] + p['b_peak'] * rho
    gaussian = np.exp(-((n - n_peak)**2) / (2 * p['sigma']**2))
    linear = 1 + p['beta'] * (n - n_peak)
    return gamma_max * gaussian * linear


def collapse_potential(n: float, rho: float, dimension: int) -> float:
    """
    Collapse potential function
    Ψ(n,ρ,D) = Γ - Γ_crit - α*(I_avg - I_sat)² - β/S_D
    """
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    i_sat = params['a'] + params['b'] * rho
    i_avg = i_avg_model(n, rho, dimension)
    
    # Coverage term
    if dimension == 3:
        gamma = gamma_model(n, rho)
        gamma_crit = max([gamma_model(n_test, rho) for n_test in np.linspace(5, 16, 50)])
    else:
        gamma = 0.1  # simplified for 4D
        gamma_crit = 1.0
    
    # Stability (simplified - use constant)
    stability = 2.0 if dimension == 4 else 1.0
    
    # Potential
    sat_penalty = COLLAPSE_PARAMS['alpha'] * (i_avg - i_sat)**2
    stab_penalty = COLLAPSE_PARAMS['beta'] / stability
    
    return gamma - gamma_crit - sat_penalty - stab_penalty


# ==================== PDE SOLVER: INFORMATION FLOW FIELD ====================

def compute_flow_field_manual(dimension: int, rho_values: List[float], 
                               n_range: Tuple[int, int]) -> Dict:
    """
    Manual implementation of Information Flow Field
    F(n,ρ,D) = ∇Ψ = (∂Ψ/∂n, ∂Ψ/∂ρ)
    """
    print(f"\n{'='*70}")
    print(f"INFORMATION FLOW FIELD SIMULATION ({dimension}D)")
    print(f"{'='*70}")
    print(f"Computing vector field F(n,ρ) = ∇Ψ")
    print(f"Density range: {rho_values}")
    print(f"Variable range: n ∈ [{n_range[0]}, {n_range[1]}]")
    
    results = {}
    
    for rho in rho_values:
        n_vals = np.linspace(n_range[0], n_range[1], 50)
        
        # Compute potential along n
        psi_vals = np.array([collapse_potential(n, rho, dimension) for n in n_vals])
        
        # Compute gradient ∂Ψ/∂n
        d_psi_dn = np.gradient(psi_vals, n_vals)
        
        # Compute ∂Ψ/∂ρ at each n
        d_psi_drho = []
        for n in n_vals:
            # Numerical derivative w.r.t. rho
            h = 0.01
            psi_plus = collapse_potential(n, rho + h, dimension)
            psi_minus = collapse_potential(n, rho - h, dimension)
            d_psi_drho.append((psi_plus - psi_minus) / (2 * h))
        d_psi_drho = np.array(d_psi_drho)
        
        # Compute divergence ∇·F ≈ ∂²Ψ/∂n² + ∂²Ψ/∂ρ²
        d2_psi_dn2 = np.gradient(d_psi_dn, n_vals)
        
        # Find critical points where ∇·F ≈ 0
        critical_indices = np.where(np.abs(d2_psi_dn2) < 0.1)[0]
        critical_n = n_vals[critical_indices] if len(critical_indices) > 0 else []
        
        results[rho] = {
            'n': n_vals,
            'psi': psi_vals,
            'd_psi_dn': d_psi_dn,
            'd_psi_drho': d_psi_drho,
            'divergence': d2_psi_dn2,
            'critical_points': critical_n
        }
        
        print(f"\nρ = {rho:.1f}:")
        print(f"  Ψ range: [{psi_vals.min():.3f}, {psi_vals.max():.3f}]")
        print(f"  ∂Ψ/∂n range: [{d_psi_dn.min():.3f}, {d_psi_dn.max():.3f}]")
        print(f"  ∇·F range: [{d2_psi_dn2.min():.3f}, {d2_psi_dn2.max():.3f}]")
        if len(critical_n) > 0:
            print(f"  Phase transition points: n ≈ {[f'{n:.1f}' for n in critical_n[:3]]}")
        else:
            print(f"  No phase transitions detected in range")
    
    return results


def compute_flow_field_fipy(dimension: int, rho: float, 
                            n_range: Tuple[int, int]) -> Dict:
    """
    FiPy implementation of Information Flow Field PDE
    Solves ∇²Ψ = source term on 2D grid (n, ρ)
    """
    if not FIPY_AVAILABLE:
        return None
    
    print(f"\nUsing FiPy PDE solver for ρ={rho}")
    
    # Create 2D mesh
    nx, ny = 50, 50
    dx, dy = (n_range[1] - n_range[0]) / nx, 0.4 / ny
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    
    # Define cell variable for Ψ
    psi = CellVariable(name="collapse potential", mesh=mesh, value=0.0)
    
    # Initial conditions based on model
    x, y = mesh.cellCenters
    n_vals = x + n_range[0]
    rho_vals = y + 0.3  # start from 0.3
    
    for i in range(len(psi)):
        psi[i] = collapse_potential(n_vals[i], rho_vals[i], dimension)
    
    # Solve Laplacian to smooth field
    eq = TransientTerm() == DiffusionTerm(coeff=0.1)
    
    # Evolve for a few timesteps
    for _ in range(5):
        eq.solve(var=psi, dt=0.1)
    
    return {'fipy_psi': psi, 'mesh': mesh}


# ==================== ODE SOLVER: DIFFERENTIAL COLLAPSE ====================

def differential_collapse_ode(state, n, rho, dimension):
    """
    ODE system for collapse dynamics
    State = [Ψ, I_avg, Γ]
    
    dΨ/dn = Γ - Γ_crit - 2α(I_avg - I_sat)·dI_avg/dn
    dI_avg/dn = λ_D * (I_sat - I_avg)  [exponential approach to saturation]
    dΓ/dn = -2(n - n_peak)/σ² * Γ  [Gaussian decline]
    """
    psi, i_avg, gamma = state
    
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    i_sat = params['a'] + params['b'] * rho
    
    # dI_avg/dn from exponential saturation model
    d_i_avg_dn = params['lambda'] * (i_sat - i_avg)
    
    # dΓ/dn from Gaussian model
    if dimension == 3:
        p = COVERAGE_PARAMS
        n_peak = p['a_peak'] + p['b_peak'] * rho
        d_gamma_dn = -2 * (n - n_peak) / (p['sigma']**2) * gamma + p['beta'] * gamma
    else:
        d_gamma_dn = -0.01 * gamma  # slow decline for 4D
    
    # Critical coverage
    if dimension == 3:
        gamma_crit = COVERAGE_PARAMS['a_max'] + COVERAGE_PARAMS['b_max'] * rho
    else:
        gamma_crit = 1.0
    
    # dΨ/dn from differential collapse equation
    d_psi_dn = gamma - gamma_crit - 2 * COLLAPSE_PARAMS['alpha'] * (i_avg - i_sat) * d_i_avg_dn
    
    return [d_psi_dn, d_i_avg_dn, d_gamma_dn]


def solve_collapse_trajectory(dimension: int, rho: float, n_range: Tuple[int, int]) -> Dict:
    """
    Solve differential collapse equation using scipy.integrate.solve_ivp
    """
    print(f"\n{'='*70}")
    print(f"DIFFERENTIAL COLLAPSE EQUATION ({dimension}D, ρ={rho})")
    print(f"{'='*70}")
    print(f"Solving ODE system: dΨ/dn = Γ - Γ_crit - 2α[I_avg - I_sat]·dI/dn")
    
    # Initial conditions at n_start
    n_start = n_range[0]
    psi_0 = collapse_potential(n_start, rho, dimension)
    i_avg_0 = i_avg_model(n_start, rho, dimension)
    gamma_0 = gamma_model(n_start, rho) if dimension == 3 else 0.1
    
    initial_state = [psi_0, i_avg_0, gamma_0]
    n_span = (n_range[0], n_range[1])
    n_eval = np.linspace(n_range[0], n_range[1], 100)
    
    print(f"\nInitial conditions at n={n_start}:")
    print(f"  Ψ₀ = {psi_0:.3f}")
    print(f"  I_avg₀ = {i_avg_0:.3f}")
    print(f"  Γ₀ = {gamma_0:.3f}")
    
    # Solve ODE
    solution = solve_ivp(
        lambda n, y: differential_collapse_ode(y, n, rho, dimension),
        n_span,
        initial_state,
        t_eval=n_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    if not solution.success:
        print(f"Warning: ODE solver did not converge")
        return None
    
    n_vals = solution.t
    psi_vals = solution.y[0]
    i_avg_vals = solution.y[1]
    gamma_vals = solution.y[2]
    
    # Compute derivatives
    d_psi_dn = np.gradient(psi_vals, n_vals)
    d_i_avg_dn = np.gradient(i_avg_vals, n_vals)
    
    # Visual analysis: observe where both dI/dn and dΨ/dn approach zero
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    i_sat = params['a'] + params['b'] * rho
    
    collapse_n = None  # Let viewer interpret visually
    
    print(f"\nODE Solution Summary:")
    print(f"  Solved for n ∈ [{n_vals[0]:.1f}, {n_vals[-1]:.1f}]")
    print(f"  Final Ψ = {psi_vals[-1]:.3f}")
    print(f"  Final I_avg = {i_avg_vals[-1]:.3f} (saturation: {i_sat:.3f})")
    print(f"  Final Γ = {gamma_vals[-1]:.3f}")
    
    return {
        'n': n_vals,
        'psi': psi_vals,
        'i_avg': i_avg_vals,
        'gamma': gamma_vals,
        'd_psi_dn': d_psi_dn,
        'd_i_avg_dn': d_i_avg_dn,
        'collapse_n': collapse_n,
        'i_sat': i_sat
    }


# ==================== VISUALIZATION ====================

def plot_flow_field(results: Dict, dimension: int, output_file: str = None):
    """Visualize information flow field"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Information Flow Field Analysis ({dimension}D)', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, (rho, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        
        # Plot 1: Collapse Potential Ψ(n)
        axes[0, 0].plot(data['n'], data['psi'], label=f'ρ={rho}', color=color, linewidth=2)
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('n (variables)')
        axes[0, 0].set_ylabel('Ψ(n,ρ)')
        axes[0, 0].set_title('Collapse Potential')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Flow component ∂Ψ/∂n
        axes[0, 1].plot(data['n'], data['d_psi_dn'], label=f'ρ={rho}', color=color, linewidth=2)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('n (variables)')
        axes[0, 1].set_ylabel('∂Ψ/∂n')
        axes[0, 1].set_title('Flow Field Component (n-direction)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Divergence ∇·F
        axes[1, 0].plot(data['n'], data['divergence'], label=f'ρ={rho}', color=color, linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].axhspan(-0.1, 0.1, alpha=0.2, color='yellow', label='Transition zone')
        axes[1, 0].set_xlabel('n (variables)')
        axes[1, 0].set_ylabel('∇·F')
        axes[1, 0].set_title('Divergence (Phase Transition Detection)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Phase portrait
        axes[1, 1].scatter(data['psi'], data['d_psi_dn'], label=f'ρ={rho}', 
                          color=color, alpha=0.6, s=20)
        if len(data['critical_points']) > 0:
            # Mark critical points
            for cp_n in data['critical_points'][:3]:
                idx = np.argmin(np.abs(data['n'] - cp_n))
                axes[1, 1].plot(data['psi'][idx], data['d_psi_dn'][idx], 
                              'k*', markersize=15, markeredgewidth=2)
    
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Ψ')
    axes[1, 1].set_ylabel('∂Ψ/∂n')
    axes[1, 1].set_title('Phase Portrait (★ = critical points)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved flow field plot to: {output_file}")
    else:
        plt.show()


def plot_collapse_trajectory(results: Dict, dimension: int, rho: float, output_file: str = None):
    """Visualize differential collapse trajectory"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Differential Collapse Trajectory ({dimension}D, ρ={rho})', 
                 fontsize=16, fontweight='bold')
    
    n = results['n']
    collapse_n = results['collapse_n']
    
    # Plot 1: State variables
    ax1 = axes[0, 0]
    ax1.plot(n, results['psi'], 'b-', linewidth=2, label='Ψ(n)')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero reference')
    ax1.set_xlabel('n (variables)')
    ax1.set_ylabel('Ψ')
    ax1.set_title('Collapse Potential Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Information density
    ax2 = axes[0, 1]
    ax2.plot(n, results['i_avg'], 'g-', linewidth=2, label='I_avg(n)')
    ax2.axhline(y=results['i_sat'], color='r', linestyle='--', alpha=0.5, label=f'I_sat={results["i_sat"]:.2f}')
    ax2.set_xlabel('n (variables)')
    ax2.set_ylabel('I_avg')
    ax2.set_title('Information Saturation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coverage Constriction
    ax3 = axes[1, 0]
    ax3.plot(n, results['gamma'], 'm-', linewidth=2, label='Γ(n)')
    ax3.set_xlabel('n (variables)')
    ax3.set_ylabel('Γ')
    ax3.set_title('Coverage Constriction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Derivatives (collapse criterion)
    ax4 = axes[1, 1]
    ax4.plot(n, results['d_psi_dn'], 'b-', linewidth=2, label='dΨ/dn')
    ax4.plot(n, results['d_i_avg_dn'], 'g-', linewidth=2, label='dI/dn')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero reference')
    ax4.set_xlabel('n (variables)')
    ax4.set_ylabel('Derivative')
    ax4.set_title('Collapse Criterion: dI/dn → 0 AND dΨ/dn → 0 (simultaneous)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved collapse trajectory plot to: {output_file}")
    else:
        plt.show()


# ==================== PDF GENERATION ====================

def create_pdf_report(output_dir: str):
    """Create comprehensive PDF report with all simulations"""
    pdf_path = os.path.join(output_dir, f"vector_calculus_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    print(f"\n{'='*70}")
    print(f"CREATING PDF REPORT: {pdf_path}")
    print(f"{'='*70}")
    
    with PdfPages(pdf_path) as pdf:
        # ========== TITLE PAGE ==========
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title_text = f"""
{'='*70}
VECTOR CALCULUS SIMULATION REPORT
Dimensional Collapse in Boolean Hypercubes
{'='*70}

Date: {datetime.now().strftime('%B %d, %Y')}
Experiment: Information Flow Field & Differential Collapse Analysis

{'='*70}
MODEL DESCRIPTION
{'='*70}

This report analyzes the vector calculus framework for dimensional
collapse in Boolean function minimization. Two main models are
implemented:

1. Information Flow Field (PDE)
   • Vector field: F(n,ρ,D) = ∇Ψ
   • Divergence analysis: ∇·F to detect phase transitions
   • Critical points where ∇·F ≈ 0

2. Differential Collapse Equation (ODE)
   • Evolution: dΨ/dn = Γ - Γ_crit - 2α[I_avg - I_sat]·dI/dn
   • Collapse criterion: dI/dn → 0 AND dΨ/dn → 0
   • Both conditions must be satisfied simultaneously
   • Trajectory analysis across variable space

{'='*70}
MATHEMATICAL FRAMEWORK
{'='*70}

Collapse Potential:
  Ψ(n,ρ,D) = Γ - Γ_crit - α(I_avg - I_sat)² - β/S_D

Information Saturation:
  I_avg = I_sat[1 - exp(-λ(n - n₀))]
  I_sat = a + bρ

Coverage Constriction (3D):
  Γ = Γ_max·exp(-(n-n_peak)²/2σ²)·(1 + β(n-n_peak))

{'='*70}
PARAMETERS
{'='*70}

3D Parameters:
  a = {PARAMS_3D['a']:.3f}, b = {PARAMS_3D['b']:.3f}
  λ = {PARAMS_3D['lambda']:.3f}, n₀ = {PARAMS_3D['n0']:.3f}

4D Parameters:
  a = {PARAMS_4D['a']:.3f}, b = {PARAMS_4D['b']:.3f}
  λ = {PARAMS_4D['lambda']:.3f}, n₀ = {PARAMS_4D['n0']:.3f}

Collapse Parameters:
  α = {COLLAPSE_PARAMS['alpha']}, β = {COLLAPSE_PARAMS['beta']}
  σ_max = {COLLAPSE_PARAMS['sigma_max']}

{'='*70}
ANALYSIS SCOPE
{'='*70}

Dimensions: 3D (n=5-16), 4D (n=11-18)
Densities: ρ ∈ {{0.3, 0.5, 0.7, 0.9}}

Simulations:
  • Flow field analysis (2 pages)
  • Collapse trajectories (8 pages, 4 densities × 2 dimensions)

{'='*70}
"""
        
        ax.text(0.5, 0.5, title_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("  ✓ Title page created")
        
        # ========== 3D SIMULATIONS ==========
        dimension = 3
        rho_values = [0.3, 0.5, 0.7, 0.9]
        n_range = (5, 16)
        
        print(f"\n  Simulating 3D (n={n_range[0]}-{n_range[1]})...")
        
        # Flow Field Analysis
        flow_results = compute_flow_field_manual(dimension, rho_values, n_range)
        
        # Create flow field figure directly for PDF
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Information Flow Field Analysis ({dimension}D)', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for idx, (rho, data) in enumerate(flow_results.items()):
            color = colors[idx % len(colors)]
            
            axes[0, 0].plot(data['n'], data['psi'], label=f'ρ={rho}', color=color, linewidth=2)
            axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0, 0].set_xlabel('n (variables)')
            axes[0, 0].set_ylabel('Ψ(n,ρ)')
            axes[0, 0].set_title('Collapse Potential')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(data['n'], data['d_psi_dn'], label=f'ρ={rho}', color=color, linewidth=2)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0, 1].set_xlabel('n (variables)')
            axes[0, 1].set_ylabel('∂Ψ/∂n')
            axes[0, 1].set_title('Flow Field Component (n-direction)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(data['n'], data['divergence'], label=f'ρ={rho}', color=color, linewidth=2)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].axhspan(-0.1, 0.1, alpha=0.2, color='yellow', label='Transition zone' if idx == 0 else '')
            axes[1, 0].set_xlabel('n (variables)')
            axes[1, 0].set_ylabel('∇·F')
            axes[1, 0].set_title('Divergence (Phase Transition Detection)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].scatter(data['psi'], data['d_psi_dn'], label=f'ρ={rho}', 
                              color=color, alpha=0.6, s=20)
            if len(data['critical_points']) > 0:
                for cp_n in data['critical_points'][:3]:
                    idx_cp = np.argmin(np.abs(data['n'] - cp_n))
                    axes[1, 1].plot(data['psi'][idx_cp], data['d_psi_dn'][idx_cp], 
                                  'k*', markersize=15, markeredgewidth=2)
        
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Ψ')
        axes[1, 1].set_ylabel('∂Ψ/∂n')
        axes[1, 1].set_title('Phase Portrait (★ = critical points)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("    ✓ 3D Flow field page added")
        
        # Collapse Trajectories for all densities
        for rho in rho_values:
            collapse_results = solve_collapse_trajectory(dimension, rho, n_range)
            
            if collapse_results:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Differential Collapse Trajectory ({dimension}D, ρ={rho})', 
                             fontsize=16, fontweight='bold')
                
                n = collapse_results['n']
                collapse_n = collapse_results['collapse_n']
                
                # Plot 1: Collapse Potential
                axes[0, 0].plot(n, collapse_results['psi'], 'b-', linewidth=2, label='Ψ(n)')
                axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero reference')
                axes[0, 0].set_xlabel('n (variables)')
                axes[0, 0].set_ylabel('Ψ')
                axes[0, 0].set_title('Collapse Potential Evolution')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Information density
                axes[0, 1].plot(n, collapse_results['i_avg'], 'g-', linewidth=2, label='I_avg(n)')
                axes[0, 1].axhline(y=collapse_results['i_sat'], color='r', linestyle='--', alpha=0.5, label=f'I_sat={collapse_results["i_sat"]:.2f}')
                axes[0, 1].set_xlabel('n (variables)')
                axes[0, 1].set_ylabel('I_avg')
                axes[0, 1].set_title('Information Saturation')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Coverage Constriction
                axes[1, 0].plot(n, collapse_results['gamma'], 'm-', linewidth=2, label='Γ(n)')
                axes[1, 0].set_xlabel('n (variables)')
                axes[1, 0].set_ylabel('Γ')
                axes[1, 0].set_title('Coverage Constriction')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Derivatives
                axes[1, 1].plot(n, collapse_results['d_psi_dn'], 'b-', linewidth=2, label='dΨ/dn')
                axes[1, 1].plot(n, collapse_results['d_i_avg_dn'], 'g-', linewidth=2, label='dI/dn')
                axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero reference')
                axes[1, 1].set_xlabel('n (variables)')
                axes[1, 1].set_ylabel('Derivative')
                axes[1, 1].set_title('Collapse Criterion: dI/dn → 0 AND dΨ/dn → 0')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"    ✓ 3D Collapse trajectory (ρ={rho}) page added")
        
        # ========== 4D SIMULATIONS ==========
        dimension = 4
        n_range = (11, 18)
        
        print(f"\n  Simulating 4D (n={n_range[0]}-{n_range[1]})...")
        
        # Flow Field Analysis
        flow_results_4d = compute_flow_field_manual(dimension, rho_values, n_range)
        
        # Create flow field figure directly for PDF
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Information Flow Field Analysis ({dimension}D)', fontsize=16, fontweight='bold')
        
        for idx, (rho, data) in enumerate(flow_results_4d.items()):
            color = colors[idx % len(colors)]
            
            axes[0, 0].plot(data['n'], data['psi'], label=f'ρ={rho}', color=color, linewidth=2)
            axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0, 0].set_xlabel('n (variables)')
            axes[0, 0].set_ylabel('Ψ(n,ρ)')
            axes[0, 0].set_title('Collapse Potential')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(data['n'], data['d_psi_dn'], label=f'ρ={rho}', color=color, linewidth=2)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[0, 1].set_xlabel('n (variables)')
            axes[0, 1].set_ylabel('∂Ψ/∂n')
            axes[0, 1].set_title('Flow Field Component (n-direction)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(data['n'], data['divergence'], label=f'ρ={rho}', color=color, linewidth=2)
            axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].axhspan(-0.1, 0.1, alpha=0.2, color='yellow', label='Transition zone' if idx == 0 else '')
            axes[1, 0].set_xlabel('n (variables)')
            axes[1, 0].set_ylabel('∇·F')
            axes[1, 0].set_title('Divergence (Phase Transition Detection)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].scatter(data['psi'], data['d_psi_dn'], label=f'ρ={rho}', 
                              color=color, alpha=0.6, s=20)
            if len(data['critical_points']) > 0:
                for cp_n in data['critical_points'][:3]:
                    idx_cp = np.argmin(np.abs(data['n'] - cp_n))
                    axes[1, 1].plot(data['psi'][idx_cp], data['d_psi_dn'][idx_cp], 
                                  'k*', markersize=15, markeredgewidth=2)
        
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Ψ')
        axes[1, 1].set_ylabel('∂Ψ/∂n')
        axes[1, 1].set_title('Phase Portrait (★ = critical points)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("    ✓ 4D Flow field page added")
        
        # Collapse Trajectories for all densities
        for rho in rho_values:
            collapse_results_4d = solve_collapse_trajectory(dimension, rho, n_range)
            
            if collapse_results_4d:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Differential Collapse Trajectory ({dimension}D, ρ={rho})', 
                             fontsize=16, fontweight='bold')
                
                n = collapse_results_4d['n']
                collapse_n = collapse_results_4d['collapse_n']
                
                # Plot 1: Collapse Potential
                axes[0, 0].plot(n, collapse_results_4d['psi'], 'b-', linewidth=2, label='Ψ(n)')
                axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero reference')
                axes[0, 0].set_xlabel('n (variables)')
                axes[0, 0].set_ylabel('Ψ')
                axes[0, 0].set_title('Collapse Potential Evolution')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Information density
                axes[0, 1].plot(n, collapse_results_4d['i_avg'], 'g-', linewidth=2, label='I_avg(n)')
                axes[0, 1].axhline(y=collapse_results_4d['i_sat'], color='r', linestyle='--', alpha=0.5, label=f'I_sat={collapse_results_4d["i_sat"]:.2f}')
                axes[0, 1].set_xlabel('n (variables)')
                axes[0, 1].set_ylabel('I_avg')
                axes[0, 1].set_title('Information Saturation')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Coverage Cnstriction
                axes[1, 0].plot(n, collapse_results_4d['gamma'], 'm-', linewidth=2, label='Γ(n)')
                axes[1, 0].set_xlabel('n (variables)')
                axes[1, 0].set_ylabel('Γ')
                axes[1, 0].set_title('Coverage Constriction')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Derivatives
                axes[1, 1].plot(n, collapse_results_4d['d_psi_dn'], 'b-', linewidth=2, label='dΨ/dn')
                axes[1, 1].plot(n, collapse_results_4d['d_i_avg_dn'], 'g-', linewidth=2, label='dI/dn')
                axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero reference')
                axes[1, 1].set_xlabel('n (variables)')
                axes[1, 1].set_ylabel('Derivative')
                axes[1, 1].set_title('Collapse Criterion: dI/dn → 0 AND dΨ/dn → 0')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                print(f"    ✓ 4D Collapse trajectory (ρ={rho}) page added")
    
    print(f"\n{'='*70}")
    print(f"✓ PDF REPORT CREATED: {pdf_path}")
    print(f"{'='*70}")
    print(f"\nTotal pages: 11 (1 title + 2 flow fields + 8 trajectories)")
    return pdf_path


# ==================== MAIN EXECUTION ====================

def main():
    """Run vector calculus simulations and generate PDF report"""
    print("="*70)
    print("VECTOR CALCULUS SIMULATION SUITE")
    print("Dimensional Collapse in Boolean Hypercubes")
    print("="*70)
    
    # Set output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "outputs", "cluster_formation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive PDF report
    pdf_path = create_pdf_report(output_dir)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nGenerated PDF report: {os.path.basename(pdf_path)}")
    print(f"Location: {output_dir}")


if __name__ == "__main__":
    main()
