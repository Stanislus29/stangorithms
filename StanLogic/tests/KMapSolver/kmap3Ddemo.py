from stanlogic.kmapsolver3D import KMapSolver3D
import random

# Example: 8-variable K-map with random pattern
print("EXAMPLE: 16-VARIABLE K-MAP (Random Pattern)")
print("="*60)
num_vars = 16

# Generate random output values (256 total for 8 variables)
# Set random seed for reproducibility
random.seed(42)
output_values_16 = [random.choice([0, 1, 'd']) for _ in range(2**num_vars)]

# Create and minimize
kmap_solver_16 = KMapSolver3D(num_vars, output_values_16)
kmap_solver_16.print_kmaps()

# Minimize the 8-variable K-map
terms, expression = kmap_solver_16.minimize_3d(form='sop')

# Generate HTML report
print("\n" + "="*60)
print("GENERATING HTML REPORT")
print("="*60)
html_file = kmap_solver_16.generate_html_report(
    filename="kmap_16var_report.html",
    form='sop',
    module_name="logic_circuit_16var"
)
print(f"✓ HTML report generated: {html_file}")
print(f"  Open this file in a web browser to view the interactive report.")