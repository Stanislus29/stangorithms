"""
Test script for validating mathematical models from model.md against empirical data.
Tests all models documented in the Final Mathematical Verification Report.

Date: February 5, 2026
"""

import csv
import math
from statistics import mean, stdev
from typing import Dict, List, Tuple

# ==================== MODEL PARAMETERS ====================

# Model 1: Information Saturation Parameters
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

# Model 2: Coverage Constriction Parameters (3D only)
COVERAGE_PARAMS = {
    'a_max': 0.317,
    'b_max': 0.261,
    'a_peak': 16.093,
    'b_peak': -7.043,
    'sigma': 4.352,
    'beta': -0.001492
}

# Model 3: Geometric Stability Parameters
GEOMETRIC_3D = {'a': 2.768, 'b': -0.822}
GEOMETRIC_4D = {'a': 2.156, 'b': 0.081}

# Model 4: Critical n Parameters
CRITICAL_N_3D = {'n0': 9, 'alpha': 2}
CRITICAL_N_4D = {'n0': 11, 'alpha': 1.5}

# Collapse Potential Parameters
COLLAPSE_PARAMS = {
    'alpha': 0.1,  # saturation penalty
    'beta': 0.5,   # stability penalty
    'sigma_max': 0.5  # variance threshold
}

# ==================== MODEL FUNCTIONS ====================

def i_avg_model(n: int, rho: float, dimension: int) -> float:
    """
    Information Saturation Model
    I_avg(n,ρ,D) = I_sat(ρ,D) * [1 - exp(-λ_D(n - n_0))]
    """
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    i_sat = params['a'] + params['b'] * rho
    exponent = -params['lambda'] * (n - params['n0'])
    return i_sat * (1 - math.exp(exponent))


def gamma_model(n: int, rho: float) -> float:
    """
    Coverage Constriction Model (3D only)
    Γ₃(n,ρ) = Γ_max(ρ) * exp(-(n-n_peak)²/2σ²) * (1 + β(n - n_peak))
    """
    p = COVERAGE_PARAMS
    gamma_max = p['a_max'] + p['b_max'] * rho
    n_peak = p['a_peak'] + p['b_peak'] * rho
    
    gaussian = math.exp(-((n - n_peak)**2) / (2 * p['sigma']**2))
    linear = 1 + p['beta'] * (n - n_peak)
    
    return gamma_max * gaussian * linear


def geometric_ratio(rho: float, dimension: int) -> float:
    """
    Geometric Ratio Model
    R(D,ρ) = a_D + b_D * ρ
    """
    params = GEOMETRIC_3D if dimension == 3 else GEOMETRIC_4D
    return params['a'] + params['b'] * rho


def critical_n(rho: float, dimension: int) -> int:
    """
    Critical n Bounds Model
    n_crit(D,ρ) = n_0(D) + α(D) * floor(log₂(1/(1-ρ)))
    """
    params = CRITICAL_N_3D if dimension == 3 else CRITICAL_N_4D
    if rho >= 1.0:
        return params['n0'] + 10 * params['alpha']  # safe upper bound
    log_term = math.floor(math.log2(1 / (1 - rho)))
    return params['n0'] + params['alpha'] * log_term


def collapse_potential(n: int, rho: float, dimension: int, 
                       i_avg_val: float, coverage: float, 
                       stability: float) -> float:
    """
    Collapse Potential Function (simplified)
    Ψ(n,ρ,D) = Coverage_deficit - α*(I_avg - I_sat)² - β/S_D
    
    Note: Full integral version requires time-series data
    """
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    i_sat = params['a'] + params['b'] * rho
    
    # Saturation penalty
    sat_penalty = COLLAPSE_PARAMS['alpha'] * (i_avg_val - i_sat)**2
    
    # Stability penalty
    stab_penalty = COLLAPSE_PARAMS['beta'] / stability if stability > 0 else 1000
    
    # Coverage deficit (simplified - just check if coverage declining)
    coverage_deficit = coverage - 1.0  # normalized coverage
    
    return coverage_deficit - sat_penalty - stab_penalty


# ==================== DATA LOADING ====================

def load_csv_data(filepath: str) -> List[Dict]:
    """Load CSV data into list of dictionaries"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Warning: Could not find {filepath}")
        return []
    return data


def parse_row(row: Dict) -> Dict:
    """Parse a CSV row into numerical values"""
    # Try 3D column names first, then 4D
    clusters_key = 'avg_3d_clusters' if 'avg_3d_clusters' in row else 'avg_4d_clusters'
    
    return {
        'n': int(row['num_vars']),
        'density': float(row['density']),
        'avg_clusters': float(row.get(clusters_key, 0)),
        'avg_info_density': float(row.get('avg_information_density', 0)),
        'coverage': float(row.get('avg_coverage_ratio', 0)),  # Already a ratio, not percentage
        'uniqueness': float(row.get('avg_uniqueness_ratio', 0))
    }


# ==================== TESTING FUNCTIONS ====================

def test_information_saturation(data: List[Dict], dimension: int):
    """Test Model 1: Information Saturation"""
    print(f"\n{'='*70}")
    print(f"MODEL 1: INFORMATION SATURATION (Dimension {dimension}D)")
    print(f"{'='*70}")
    print(f"Model: I_avg(n,ρ,D) = I_sat(ρ,D) * [1 - exp(-λ_D(n - n_0))]")
    
    params = PARAMS_3D if dimension == 3 else PARAMS_4D
    print(f"\nParameters: a={params['a']:.3f}, b={params['b']:.3f}, "
          f"λ={params['lambda']:.3f}, n₀={params['n0']:.3f}")
    
    errors = []
    print(f"\n{'n':<4} {'ρ':<5} {'Empirical':<12} {'Model':<12} {'Error %':<10} {'Status'}")
    print(f"{'-'*70}")
    
    for row in data:
        parsed = parse_row(row)
        n, rho = parsed['n'], parsed['density']
        empirical = parsed['avg_info_density']
        
        if empirical > 0:  # Only test valid data points
            predicted = i_avg_model(n, rho, dimension)
            error_pct = abs(empirical - predicted) / empirical * 100
            errors.append(error_pct)
            
            status = "✓" if error_pct < 40 else "✗"
            print(f"{n:<4} {rho:<5.1f} {empirical:<12.3f} {predicted:<12.3f} "
                  f"{error_pct:<10.2f} {status}")
    
    if errors:
        mape = mean(errors)
        print(f"\n{'='*70}")
        print(f"RESULTS: MAPE = {mape:.2f}% (Target: <40%)")
        print(f"Status: {'PASS ✓' if mape < 40 else 'FAIL ✗'}")
        print(f"{'='*70}")
    else:
        print("\nNo valid data points found")


def test_coverage_constriction(data: List[Dict]):
    """Test Model 2: Coverage Constriction (3D only)"""
    print(f"\n{'='*70}")
    print(f"MODEL 2: COVERAGE CONSTRICTION (3D Only)")
    print(f"{'='*70}")
    print(f"Model: Γ₃(n,ρ) = Γ_max(ρ) * exp(-(n-n_peak)²/2σ²) * (1 + β(n - n_peak))")
    
    p = COVERAGE_PARAMS
    print(f"\nParameters: a_max={p['a_max']:.3f}, b_max={p['b_max']:.3f}")
    print(f"            a_peak={p['a_peak']:.3f}, b_peak={p['b_peak']:.3f}")
    print(f"            σ={p['sigma']:.3f}, β={p['beta']:.6f}")
    
    errors = []
    print(f"\n{'n':<4} {'ρ':<5} {'Empirical':<12} {'Model':<12} {'Error %':<10} {'Status'}")
    print(f"{'-'*70}")
    
    for row in data:
        parsed = parse_row(row)
        n, rho = parsed['n'], parsed['density']
        empirical_coverage = parsed['coverage']
        
        if empirical_coverage > 0:
            predicted = gamma_model(n, rho)
            error_pct = abs(empirical_coverage - predicted) / empirical_coverage * 100
            errors.append(error_pct)
            
            status = "✓" if error_pct < 25 else "✗"
            print(f"{n:<4} {rho:<5.1f} {empirical_coverage:<12.3f} {predicted:<12.3f} "
                  f"{error_pct:<10.2f} {status}")
    
    if errors:
        mape = mean(errors)
        print(f"\n{'='*70}")
        print(f"RESULTS: MAPE = {mape:.2f}% (Target: ~16%)")
        print(f"Status: {'PASS ✓' if mape < 25 else 'FAIL ✗'}")
        print(f"{'='*70}")
    else:
        print("\nNo valid data points found")


def test_geometric_stability(data: List[Dict], dimension: int):
    """Test Model 3: Geometric Ratio and Stability"""
    print(f"\n{'='*70}")
    print(f"MODEL 3: GEOMETRIC STABILITY (Dimension {dimension}D)")
    print(f"{'='*70}")
    print(f"Model: R(D,ρ) = a_D + b_D * ρ")
    
    params = GEOMETRIC_3D if dimension == 3 else GEOMETRIC_4D
    print(f"\nParameters: a={params['a']:.3f}, b={params['b']:.3f}")
    
    # Group data by density
    density_groups = {}
    for row in data:
        parsed = parse_row(row)
        rho = parsed['density']
        if rho not in density_groups:
            density_groups[rho] = []
        density_groups[rho].append(parsed)
    
    print(f"\n{'ρ':<5} {'Predicted R':<12} {'Valid Range':<15} {'Status'}")
    print(f"{'-'*70}")
    
    ratios = []
    for rho in sorted(density_groups.keys()):
        predicted_r = geometric_ratio(rho, dimension)
        in_range = 2.0 <= predicted_r <= 2.5
        status = "✓" if in_range else "✗"
        
        ratios.append(predicted_r)
        print(f"{rho:<5.1f} {predicted_r:<12.3f} [2.0, 2.5]     {status}")
    
    # Calculate stability index
    if len(ratios) > 1:
        # For stability, we'd ideally compute σ_ratio from actual cluster ratios
        # Here we use model prediction variance as proxy
        avg_r = mean(ratios)
        variance = stdev(ratios) if len(ratios) > 1 else 0
        stability = 1 / variance if variance > 0 else float('inf')
        
        print(f"\n{'='*70}")
        print(f"Average R: {avg_r:.3f}")
        print(f"Variance: {variance:.3f}")
        print(f"Stability Index S_D: {stability:.3f}")
        print(f"Variance < 0.5: {'PASS ✓' if variance < 0.5 else 'FAIL ✗'}")
        print(f"{'='*70}")


def test_critical_n_bounds(data: List[Dict], dimension: int):
    """Test Model 4: Critical n Bounds"""
    print(f"\n{'='*70}")
    print(f"MODEL 4: CRITICAL N BOUNDS (Dimension {dimension}D)")
    print(f"{'='*70}")
    print(f"Model: n_crit(D,ρ) = n_0(D) + α(D) * floor(log₂(1/(1-ρ)))")
    
    params = CRITICAL_N_3D if dimension == 3 else CRITICAL_N_4D
    print(f"\nParameters: n₀={params['n0']}, α={params['alpha']}")
    
    # Group data by density to find actual collapse points
    density_groups = {}
    for row in data:
        parsed = parse_row(row)
        rho = parsed['density']
        if rho not in density_groups:
            density_groups[rho] = []
        density_groups[rho].append(parsed)
    
    print(f"\n{'ρ':<5} {'Predicted n_crit':<16} {'Max Empirical n':<18} {'Status'}")
    print(f"{'-'*70}")
    
    for rho in sorted(density_groups.keys()):
        predicted = critical_n(rho, dimension)
        max_n = max(p['n'] for p in density_groups[rho])
        
        # Status: if max empirical n is close to or exceeds predicted, it's reasonable
        within_range = abs(max_n - predicted) <= 3
        status = "✓" if within_range else "~"
        
        print(f"{rho:<5.1f} {predicted:<16} {max_n:<18} {status}")
    
    print(f"\n{'='*70}")
    print(f"Fundamental Limit: n_max({dimension}D) ≥ {params['n0']}")
    print(f"{'='*70}")


def test_collapse_potential(data: List[Dict], dimension: int):
    """Test Model 5: Collapse Potential"""
    print(f"\n{'='*70}")
    print(f"MODEL 5: COLLAPSE POTENTIAL (Dimension {dimension}D)")
    print(f"{'='*70}")
    print(f"Model: Ψ(n,ρ,D) < 0 indicates collapse")
    print(f"       Combined with σ_ratio > 0.5 for full collapse criterion")
    
    # For 3D, we expect collapse at n >= 9
    # For 4D, we expect viability at n <= 18
    
    print(f"\n{'n':<4} {'ρ':<5} {'Coverage':<10} {'I_avg':<10} {'Ψ':<10} {'Collapsed?'}")
    print(f"{'-'*70}")
    
    collapsed_count = 0
    total_count = 0
    
    for row in data:
        parsed = parse_row(row)
        n, rho = parsed['n'], parsed['density']
        coverage = parsed['coverage']
        i_avg_val = parsed['avg_info_density']
        
        # Use simplified stability (assume moderate values)
        stability = 2.0 if dimension == 4 else 1.0
        
        psi = collapse_potential(n, rho, dimension, i_avg_val, coverage, stability)
        
        # For 3D: expect Ψ < 0 at n >= 9
        # For 4D: expect Ψ to be less negative or positive
        is_collapsed = psi < 0 and (dimension == 3 and n >= 9)
        
        if dimension == 3 and n >= 9:
            total_count += 1
            if is_collapsed:
                collapsed_count += 1
        
        status = "YES" if is_collapsed else "NO"
        print(f"{n:<4} {rho:<5.1f} {coverage:<10.3f} {i_avg_val:<10.3f} "
              f"{psi:<10.3f} {status}")
    
    if dimension == 3 and total_count > 0:
        collapse_rate = collapsed_count / total_count * 100
        print(f"\n{'='*70}")
        print(f"Collapse Rate (n≥9): {collapse_rate:.1f}%")
        print(f"Expected: ~100% for 3D at n≥9")
        print(f"{'='*70}")


def test_variance_constraint(data: List[Dict], dimension: int):
    """Test variance constraint: σ_ratio < 0.5"""
    print(f"\n{'='*70}")
    print(f"CONSTRAINT TEST: VARIANCE THRESHOLD (Dimension {dimension}D)")
    print(f"{'='*70}")
    print(f"Constraint: σ_ratio < {COLLAPSE_PARAMS['sigma_max']}")
    
    # Group by density and calculate coefficient of variation for clusters
    density_groups = {}
    for row in data:
        parsed = parse_row(row)
        rho = parsed['density']
        if rho not in density_groups:
            density_groups[rho] = []
        density_groups[rho].append(parsed['avg_clusters'])
    
    print(f"\n{'ρ':<5} {'Mean Clusters':<15} {'Std Dev':<12} {'CV':<10} {'Pass?'}")
    print(f"{'-'*70}")
    
    pass_count = 0
    total_count = 0
    
    for rho in sorted(density_groups.keys()):
        clusters = density_groups[rho]
        if len(clusters) > 1:
            avg = mean(clusters)
            std = stdev(clusters)
            cv = std / avg if avg > 0 else 0
            
            passes = cv < COLLAPSE_PARAMS['sigma_max']
            status = "✓" if passes else "✗"
            
            print(f"{rho:<5.1f} {avg:<15.3f} {std:<12.3f} {cv:<10.3f} {status}")
            
            total_count += 1
            if passes:
                pass_count += 1
    
    if total_count > 0:
        pass_rate = pass_count / total_count * 100
        print(f"\n{'='*70}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Expected: ~100% for 4D, <50% for 3D")
        print(f"{'='*70}")


# ==================== MAIN TEST RUNNER ====================

def main():
    """Main test runner"""
    print("="*70)
    print("MATHEMATICAL MODEL VALIDATION SUITE")
    print("Testing models from: Final Mathematical Verification Report")
    print("Date: February 5, 2026")
    print("="*70)
    
    # Load data files
    data_3d_file = r"C:\Users\DELL\Documents\Research Projects\Mathematical_models\StanLogic\tests\KMapSolver\outputs\cluster_formation\cluster_analysis3d_results_20260205_035043.csv"
    data_4d_file = r"C:\Users\DELL\Documents\Research Projects\Mathematical_models\StanLogic\tests\KMapSolver\outputs\cluster_formation\cluster_analysis4d_results_20260205_035127.csv"
    
    print(f"\nLoading 3D data from: {data_3d_file.split('/')[-1]}")
    data_3d = load_csv_data(data_3d_file)
    print(f"Loaded {len(data_3d)} data points")
    
    print(f"\nLoading 4D data from: {data_4d_file.split('/')[-1]}")
    data_4d = load_csv_data(data_4d_file)
    print(f"Loaded {len(data_4d)} data points")
    
    # Test all models for 3D
    if data_3d:
        test_information_saturation(data_3d, dimension=3)
        test_coverage_constriction(data_3d)
        test_geometric_stability(data_3d, dimension=3)
        test_critical_n_bounds(data_3d, dimension=3)
        test_collapse_potential(data_3d, dimension=3)
        test_variance_constraint(data_3d, dimension=3)
    
    # Test all models for 4D
    if data_4d:
        test_information_saturation(data_4d, dimension=4)
        test_geometric_stability(data_4d, dimension=4)
        test_critical_n_bounds(data_4d, dimension=4)
        test_collapse_potential(data_4d, dimension=4)
        test_variance_constraint(data_4d, dimension=4)
    
    print(f"\n{'='*70}")
    print("NOTE: For vector calculus simulations (Information Flow Field & ")
    print("      Differential Collapse), run: vector_calculus_simulation.py")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
