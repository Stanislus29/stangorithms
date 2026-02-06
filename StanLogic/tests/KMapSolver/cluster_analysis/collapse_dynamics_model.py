"""
Collapse Dynamics Model - Automated Derivation
Analyzes 3D and 4D cluster formation data to derive optimal collapse prediction model
"""

import csv
import os
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "cluster_formation")

CSV_3D = os.path.join(OUTPUT_DIR, "cluster_analysis3d_results_20260202_180924.csv")
CSV_4D_1 = os.path.join(OUTPUT_DIR, "cluster_analysis4d_results_20260203_132052.csv")
CSV_4D_2 = os.path.join(OUTPUT_DIR, "cluster_analysis4d_results_20260203_141105.csv")

# Collapse data (from previous analysis)
COLLAPSE_DIR = OUTPUT_DIR
COLLAPSE_FILES = {
    0.3: os.path.join(COLLAPSE_DIR, "4d_collapse_density_0.3_analysis.csv"),
    0.5: os.path.join(COLLAPSE_DIR, "4d_collapse_density_0.5_analysis.csv"),
    0.7: os.path.join(COLLAPSE_DIR, "4d_collapse_density_0.7_analysis.csv"),
    0.9: os.path.join(COLLAPSE_DIR, "4d_collapse_density_0.9_analysis.csv")
}

print("="*80)
print("COLLAPSE DYNAMICS MODEL - AUTOMATED DERIVATION")
print("="*80)
print(f"\nAnalyzing data from:")
print(f"  3D: {CSV_3D}")
print(f"  4D: {CSV_4D_1}")
print(f"  4D: {CSV_4D_2}")
print()

def read_cluster_data(csv_path, dimension):
    """Read cluster analysis data."""
    data_by_density = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            density = float(row['density'])
            num_vars = int(row['num_vars'])
            
            if density not in data_by_density:
                data_by_density[density] = {
                    'n': [],
                    'clusters': [],
                    'info_density': [],
                    'coverage': [],
                    'uniqueness_avg': [],
                    'uniqueness_min': [],
                    'uniqueness_max': [],
                    'dimension': dimension
                }
            
            cluster_key = f'avg_{dimension.lower()}_clusters'
            data_by_density[density]['n'].append(num_vars)
            data_by_density[density]['clusters'].append(float(row[cluster_key]))
            data_by_density[density]['info_density'].append(float(row['avg_information_density']))
            data_by_density[density]['coverage'].append(float(row['avg_coverage_ratio']))
            data_by_density[density]['uniqueness_avg'].append(float(row['avg_uniqueness_ratio']))
            data_by_density[density]['uniqueness_min'].append(float(row['avg_min_uniqueness_ratio']))
            data_by_density[density]['uniqueness_max'].append(float(row['avg_max_uniqueness_ratio']))
    
    return data_by_density

def read_collapse_data():
    """Read collapse point data."""
    collapse_points = {}
    
    for density, filepath in COLLAPSE_FILES.items():
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if float(row['raw_coverage']) >= 100.0:
                        collapse_points[density] = int(row['n'])
                        break
    
    return collapse_points

# Read all data
print("Reading data...")
data_3d = read_cluster_data(CSV_3D, '3D')
data_4d_1 = read_cluster_data(CSV_4D_1, '4D')
data_4d_2 = read_cluster_data(CSV_4D_2, '4D')

# Merge 4D datasets (average if overlapping)
data_4d = {}
for density in set(list(data_4d_1.keys()) + list(data_4d_2.keys())):
    if density in data_4d_1 and density in data_4d_2:
        # Merge by averaging matching n values
        all_n = sorted(set(data_4d_1[density]['n'] + data_4d_2[density]['n']))
        data_4d[density] = {
            'n': all_n,
            'clusters': [],
            'info_density': [],
            'coverage': [],
            'uniqueness_avg': [],
            'uniqueness_min': [],
            'uniqueness_max': [],
            'dimension': '4D'
        }
        
        for n in all_n:
            values = {key: [] for key in ['clusters', 'info_density', 'coverage', 
                                         'uniqueness_avg', 'uniqueness_min', 'uniqueness_max']}
            
            if n in data_4d_1[density]['n']:
                idx = data_4d_1[density]['n'].index(n)
                for key in values.keys():
                    values[key].append(data_4d_1[density][key][idx])
            
            if n in data_4d_2[density]['n']:
                idx = data_4d_2[density]['n'].index(n)
                for key in values.keys():
                    values[key].append(data_4d_2[density][key][idx])
            
            for key in values.keys():
                data_4d[density][key].append(np.mean(values[key]))
    elif density in data_4d_1:
        data_4d[density] = data_4d_1[density]
    else:
        data_4d[density] = data_4d_2[density]

collapse_points = read_collapse_data()

print(f"Loaded 3D data for {len(data_3d)} densities")
print(f"Loaded 4D data for {len(data_4d)} densities")
print(f"Collapse points identified: {collapse_points}")
print()

# PHASE 1: CORRELATION ANALYSIS
print("="*80)
print("PHASE 1: CORRELATION ANALYSIS")
print("="*80)

def compute_correlations(data_dict):
    """Compute correlations between all variables."""
    correlations = {}
    
    for density, data in data_dict.items():
        n = np.array(data['n'])
        clusters = np.array(data['clusters'])
        info_density = np.array(data['info_density'])
        coverage = np.array(data['coverage'])
        uniqueness_avg = np.array(data['uniqueness_avg'])
        uniqueness_min = np.array(data['uniqueness_min'])
        uniqueness_max = np.array(data['uniqueness_max'])
        
        # Derived metrics
        uniqueness_variance = uniqueness_max - uniqueness_min
        uniqueness_efficiency = uniqueness_avg / np.where(uniqueness_variance > 0, uniqueness_variance, 1)
        coverage_rate = np.gradient(coverage)
        cluster_rate = np.gradient(np.log(clusters)) if len(clusters) > 1 else np.zeros_like(clusters)
        
        # Compute key correlations
        correlations[density] = {
            'coverage_vs_uniqueness': stats.pearsonr(coverage, uniqueness_avg)[0] if len(coverage) > 2 else 0,
            'coverage_vs_variance': stats.pearsonr(coverage, uniqueness_variance)[0] if len(coverage) > 2 else 0,
            'coverage_vs_efficiency': stats.pearsonr(coverage, uniqueness_efficiency)[0] if len(coverage) > 2 else 0,
            'uniqueness_vs_n': stats.pearsonr(n, uniqueness_avg)[0] if len(n) > 2 else 0,
            'variance_vs_n': stats.pearsonr(n, uniqueness_variance)[0] if len(n) > 2 else 0,
            'info_density_vs_n': stats.pearsonr(n, info_density)[0] if len(n) > 2 else 0,
            'cluster_growth_rate': np.mean(cluster_rate) if len(cluster_rate) > 0 else 0
        }
    
    return correlations

corr_3d = compute_correlations(data_3d)
corr_4d = compute_correlations(data_4d)

print("\n3D Correlations (averaged across densities):")
if corr_3d:
    avg_corr = {key: np.mean([corr_3d[d][key] for d in corr_3d]) for key in corr_3d[list(corr_3d.keys())[0]].keys()}
    for key, val in avg_corr.items():
        print(f"  {key}: {val:.4f}")

print("\n4D Correlations (averaged across densities):")
if corr_4d:
    avg_corr = {key: np.mean([corr_4d[d][key] for d in corr_4d]) for key in corr_4d[list(corr_4d.keys())[0]].keys()}
    for key, val in avg_corr.items():
        print(f"  {key}: {val:.4f}")

# PHASE 2: IDENTIFY KEY PATTERNS
print("\n" + "="*80)
print("PHASE 2: PATTERN IDENTIFICATION")
print("="*80)

patterns = {}

for density in sorted(data_4d.keys()):
    data = data_4d[density]
    n = np.array(data['n'])
    uniqueness_avg = np.array(data['uniqueness_avg'])
    uniqueness_variance = np.array(data['uniqueness_max']) - np.array(data['uniqueness_min'])
    coverage = np.array(data['coverage'])
    
    # Pattern 1: Uniqueness progression
    if len(uniqueness_avg) > 1:
        uniqueness_diffs = np.diff(uniqueness_avg)
        avg_diff = np.mean(uniqueness_diffs)
        std_diff = np.std(uniqueness_diffs)
        is_arithmetic = std_diff < abs(avg_diff) * 0.15
    else:
        avg_diff = 0
        is_arithmetic = False
    
    # Pattern 2: Variance growth
    if len(uniqueness_variance) > 1:
        variance_slope = np.polyfit(n, uniqueness_variance, 1)[0]
    else:
        variance_slope = 0
    
    # Pattern 3: Coverage saturation
    if len(coverage) > 2:
        coverage_accel = np.mean(np.diff(np.diff(coverage)))
    else:
        coverage_accel = 0
    
    patterns[density] = {
        'uniqueness_diff': avg_diff,
        'uniqueness_arithmetic': is_arithmetic,
        'variance_slope': variance_slope,
        'coverage_accel': coverage_accel,
        'collapse_n': collapse_points.get(density, None)
    }
    
    print(f"\nDensity {density}:")
    print(f"  Uniqueness progression: d = {avg_diff:.6f} ({'Arithmetic' if is_arithmetic else 'Variable'})")
    print(f"  Variance growth rate: {variance_slope:.6f} per n")
    print(f"  Coverage acceleration: {coverage_accel:.6f}")
    print(f"  Collapse point: {patterns[density]['collapse_n']}")

# PHASE 3: MODEL DERIVATION
print("\n" + "="*80)
print("PHASE 3: MODEL DERIVATION")
print("="*80)

# Candidate models
candidate_models = {}

# Model 1: Uniqueness Threshold Model
# Hypothesis: Collapse when uniqueness drops below threshold relative to variance
print("\nModel 1: Uniqueness-Variance Threshold")
print("-" * 80)

collapse_scores_m1 = []
for density, pattern in patterns.items():
    if pattern['collapse_n'] is not None:
        data = data_4d[density]
        
        # Find closest n value (may need interpolation)
        closest_idx = min(range(len(data['n'])), 
                         key=lambda i: abs(data['n'][i] - pattern['collapse_n']))
        
        u_avg = data['uniqueness_avg'][closest_idx]
        u_var = data['uniqueness_max'][closest_idx] - data['uniqueness_min'][closest_idx]
        score = u_avg / u_var if u_var > 0 else float('inf')
        collapse_scores_m1.append(score)
        print(f"  Density {density}: U_avg/U_var at collapse (n≈{data['n'][closest_idx]}) = {score:.4f}")

if collapse_scores_m1:
    threshold_m1 = np.mean(collapse_scores_m1)
    std_m1 = np.std(collapse_scores_m1)
    print(f"\n  Collapse threshold: {threshold_m1:.4f} ± {std_m1:.4f}")
    print(f"  Model: Collapse when U_avg / (U_max - U_min) < {threshold_m1:.4f}")

# Model 2: Weighted Distribution Efficiency
# Hypothesis: Collapse when distribution becomes too unbalanced
print("\n\nModel 2: Distribution Efficiency Score")
print("-" * 80)

collapse_scores_m2 = []
for density, pattern in patterns.items():
    if pattern['collapse_n'] is not None:
        data = data_4d[density]
        
        # Find closest n value
        closest_idx = min(range(len(data['n'])), 
                         key=lambda i: abs(data['n'][i] - pattern['collapse_n']))
        
        u_avg = data['uniqueness_avg'][closest_idx]
        u_min = data['uniqueness_min'][closest_idx]
        u_max = data['uniqueness_max'][closest_idx]
        
        # Distribution efficiency: penalize both low average AND high spread
        efficiency = u_avg * (1 - (u_max - u_min) / (u_max + 0.01))
        collapse_scores_m2.append(efficiency)
        print(f"  Density {density}: Efficiency at collapse (n≈{data['n'][closest_idx]}) = {efficiency:.4f}")

if collapse_scores_m2:
    threshold_m2 = np.mean(collapse_scores_m2)
    std_m2 = np.std(collapse_scores_m2)
    print(f"\n  Collapse threshold: {threshold_m2:.4f} ± {std_m2:.4f}")
    print(f"  Model: Collapse when U_avg × (1 - (U_max - U_min) / U_max) < {threshold_m2:.4f}")

# Model 3: Phase Space Trajectory
# Hypothesis: Collapse occurs at critical point in (U_avg, variance) space
print("\n\nModel 3: Phase Space Critical Boundary")
print("-" * 80)

collapse_coords = []
for density, pattern in patterns.items():
    if pattern['collapse_n'] is not None:
        data = data_4d[density]
        
        # Find closest n value
        closest_idx = min(range(len(data['n'])), 
                         key=lambda i: abs(data['n'][i] - pattern['collapse_n']))
        
        u_avg = data['uniqueness_avg'][closest_idx]
        u_var = data['uniqueness_max'][closest_idx] - data['uniqueness_min'][closest_idx]
        collapse_coords.append((u_avg, u_var))
        print(f"  Density {density}: (U_avg, U_var) at collapse (n≈{data['n'][closest_idx]}) = ({u_avg:.4f}, {u_var:.4f})")

if len(collapse_coords) >= 2:
    # Fit linear boundary: U_avg = m * U_var + b
    u_avg_vals = [c[0] for c in collapse_coords]
    u_var_vals = [c[1] for c in collapse_coords]
    
    slope, intercept = np.polyfit(u_var_vals, u_avg_vals, 1)
    r_squared = np.corrcoef(u_var_vals, u_avg_vals)[0, 1]**2
    
    print(f"\n  Boundary fit: U_avg = {slope:.4f} × U_var + {intercept:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Model: Collapse when U_avg < {slope:.4f} × (U_max - U_min) + {intercept:.4f}")

# Model 4: Composite Stability Index
# Hypothesis: Combine coverage pressure and distribution quality
print("\n\nModel 4: Composite Stability Index")
print("-" * 80)

collapse_scores_m4 = []
for density, pattern in patterns.items():
    if pattern['collapse_n'] is not None:
        data = data_4d[density]
        
        # Find closest n value
        closest_idx = min(range(len(data['n'])), 
                         key=lambda i: abs(data['n'][i] - pattern['collapse_n']))
        
        coverage = data['coverage'][closest_idx]
        u_avg = data['uniqueness_avg'][closest_idx]
        u_var = data['uniqueness_max'][closest_idx] - data['uniqueness_min'][closest_idx]
        
        # Stability = uniqueness quality - coverage pressure
        stability = u_avg / (1 + u_var) - coverage
        collapse_scores_m4.append(stability)
        print(f"  Density {density}: Stability at collapse (n≈{data['n'][closest_idx]}) = {stability:.4f}")

if collapse_scores_m4:
    threshold_m4 = np.mean(collapse_scores_m4)
    std_m4 = np.std(collapse_scores_m4)
    print(f"\n  Collapse threshold: {threshold_m4:.4f} ± {std_m4:.4f}")
    print(f"  Model: Collapse when [U_avg / (1 + U_var) - Coverage] < {threshold_m4:.4f}")

# PHASE 4: MODEL VALIDATION
print("\n" + "="*80)
print("PHASE 4: MODEL VALIDATION")
print("="*80)

def predict_collapse_m1(u_avg, u_var, threshold):
    """Model 1: Ratio threshold."""
    return u_avg / u_var if u_var > 0 else float('inf')

def predict_collapse_m2(u_avg, u_min, u_max, threshold):
    """Model 2: Efficiency score."""
    return u_avg * (1 - (u_max - u_min) / (u_max + 0.01))

def predict_collapse_m3(u_avg, u_var, slope, intercept):
    """Model 3: Phase space boundary."""
    boundary = slope * u_var + intercept
    return u_avg - boundary

def predict_collapse_m4(u_avg, u_var, coverage, threshold):
    """Model 4: Composite stability."""
    return u_avg / (1 + u_var) - coverage

# Validate on all densities (including non-collapsed)
validation_results = {
    'Model 1': {'correct': 0, 'total': 0, 'scores': []},
    'Model 2': {'correct': 0, 'total': 0, 'scores': []},
    'Model 3': {'correct': 0, 'total': 0, 'scores': []},
    'Model 4': {'correct': 0, 'total': 0, 'scores': []}
}

for density in sorted(data_4d.keys()):
    data = data_4d[density]
    true_collapse = collapse_points.get(density, None)
    
    for i, n in enumerate(data['n']):
        u_avg = data['uniqueness_avg'][i]
        u_min = data['uniqueness_min'][i]
        u_max = data['uniqueness_max'][i]
        u_var = u_max - u_min
        coverage = data['coverage'][i]
        
        actually_collapsed = true_collapse is not None and n >= true_collapse
        
        # Model 1
        if collapse_scores_m1:
            score1 = predict_collapse_m1(u_avg, u_var, threshold_m1)
            predicted_collapse1 = score1 < threshold_m1
            validation_results['Model 1']['total'] += 1
            if predicted_collapse1 == actually_collapsed:
                validation_results['Model 1']['correct'] += 1
            validation_results['Model 1']['scores'].append(abs(score1 - threshold_m1))
        
        # Model 2
        if collapse_scores_m2:
            score2 = predict_collapse_m2(u_avg, u_min, u_max, threshold_m2)
            predicted_collapse2 = score2 < threshold_m2
            validation_results['Model 2']['total'] += 1
            if predicted_collapse2 == actually_collapsed:
                validation_results['Model 2']['correct'] += 1
            validation_results['Model 2']['scores'].append(abs(score2 - threshold_m2))
        
        # Model 3
        if len(collapse_coords) >= 2:
            score3 = predict_collapse_m3(u_avg, u_var, slope, intercept)
            predicted_collapse3 = score3 < 0
            validation_results['Model 3']['total'] += 1
            if predicted_collapse3 == actually_collapsed:
                validation_results['Model 3']['correct'] += 1
            validation_results['Model 3']['scores'].append(abs(score3))
        
        # Model 4
        if collapse_scores_m4:
            score4 = predict_collapse_m4(u_avg, u_var, coverage, threshold_m4)
            predicted_collapse4 = score4 < threshold_m4
            validation_results['Model 4']['total'] += 1
            if predicted_collapse4 == actually_collapsed:
                validation_results['Model 4']['correct'] += 1
            validation_results['Model 4']['scores'].append(abs(score4 - threshold_m4))

# Report validation results
print("\nAccuracy on all data points:")
for model, results in validation_results.items():
    if results['total'] > 0:
        accuracy = results['correct'] / results['total']
        avg_error = np.mean(results['scores'])
        print(f"  {model}: {accuracy:.2%} ({results['correct']}/{results['total']}) | Avg error: {avg_error:.4f}")

# Select best model
if any(r['total'] > 0 for r in validation_results.values()):
    best_model = max(validation_results.items(), 
                    key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0)

    print("\n" + "="*80)
    print("RECOMMENDED MODEL")
    print("="*80)
    print(f"\nBest performing model: {best_model[0]}")
    if best_model[1]['total'] > 0:
        print(f"Accuracy: {best_model[1]['correct']/best_model[1]['total']:.2%}")

    if best_model[0] == 'Model 1' and collapse_scores_m1:
        print(f"\nCollapse Criterion:")
        print(f"  U_avg / (U_max - U_min) < {threshold_m1:.4f}")
        print(f"\nInterpretation:")
        print(f"  System collapses when average uniqueness is less than {threshold_m1:.4f}× the uniqueness spread")
    elif best_model[0] == 'Model 2' and collapse_scores_m2:
        print(f"\nCollapse Criterion:")
        print(f"  U_avg × (1 - (U_max - U_min) / U_max) < {threshold_m2:.4f}")
        print(f"\nInterpretation:")
        print(f"  System collapses when distribution efficiency falls below {threshold_m2:.4f}")
    elif best_model[0] == 'Model 3' and len(collapse_coords) >= 2:
        print(f"\nCollapse Criterion:")
        print(f"  U_avg < {slope:.4f} × (U_max - U_min) + {intercept:.4f}")
        print(f"\nInterpretation:")
        print(f"  System collapses when uniqueness trajectory crosses critical boundary in phase space")
    elif best_model[0] == 'Model 4' and collapse_scores_m4:
        print(f"\nCollapse Criterion:")
        print(f"  [U_avg / (1 + U_var)] - Coverage < {threshold_m4:.4f}")
        print(f"\nInterpretation:")
        print(f"  System collapses when stability index falls below {threshold_m4:.4f}")
else:
    print("\n" + "="*80)
    print("INSUFFICIENT DATA FOR MODEL VALIDATION")
    print("="*80)
    print("\nNo validation could be performed. Need collapse points within data range.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
