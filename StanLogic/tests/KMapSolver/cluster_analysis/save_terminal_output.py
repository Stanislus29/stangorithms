import csv
import os
from datetime import datetime

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "cluster_formation")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = os.path.join(OUTPUTS_DIR, f"terminal_output_4d_{timestamp}.csv")

# Data from terminal output (removing duplicates)
data = [
    {'Vars': 16, 'Density': 0.9, 'Avg_Clusters': 460.00, 'Avg_Info_Density': 33.53, 'Avg_Coverage': 20.52},
    {'Vars': 17, 'Density': 0.9, 'Avg_Clusters': 953.00, 'Avg_Info_Density': 33.75, 'Avg_Coverage': 21.29},
    {'Vars': 18, 'Density': 0.9, 'Avg_Clusters': 2283.00, 'Avg_Info_Density': 33.47, 'Avg_Coverage': 23.34},
    {'Vars': 19, 'Density': 0.9, 'Avg_Clusters': 4755.00, 'Avg_Info_Density': 33.66, 'Avg_Coverage': 24.38},
]

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ['Vars', 'Density', 'Avg_Clusters', 'Avg_Info_Density', 'Avg_Coverage']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f"✓ Data written to: {OUTPUT_CSV}")
print(f"  Total rows: {len(data)} (duplicates removed)")
print(f"\nData summary:")
print(f"{'Vars':<6} {'Density':<10} {'Avg Clusters':<14} {'Avg Info Density':<18} {'Avg Coverage':<14}")
print("-" * 70)
for row in data:
    print(f"{row['Vars']:<6} {row['Density']:<10.1f} {row['Avg_Clusters']:<14.2f} {row['Avg_Info_Density']:<18.2f} {row['Avg_Coverage']:<14.2f}%")
