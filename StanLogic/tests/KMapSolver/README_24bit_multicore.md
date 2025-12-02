# 24-Bit Multicore Boolean Minimization Test

This test demonstrates high-performance Boolean minimization for 24-bit functions using parallel processing across 4 CPU cores.

## Overview

- **Total Variables**: 24
- **Total Chunks**: 256 (2^8)
- **K-map Variables per Chunk**: 16
- **Chunk Selector Variables**: 8
- **CPU Cores Used**: 4
- **Chunks per Core**: 64

## Architecture

The test uses the new `KMapSolver4D` class which implements chunked processing:

1. **Chunk Distribution**: Divides 256 chunks equally across 4 cores (64 chunks each)
2. **Parallel Processing**: Each core processes its chunk range independently
3. **CSV Output**: Each core writes essential prime implicants to its own CSV file
4. **Merging**: After completion, all CSV files are merged into one
5. **Final Union**: Bitwise union can be performed on merged results

## Performance Estimates

Based on observed processing time of ~76 seconds per chunk:

- **Sequential Processing**: 256 × 76s = **5.4 hours**
- **4-Core Parallel**: 64 × 76s = **1.3 hours** (wall-clock time)
- **Speedup**: ~4x faster

## Usage

### Basic Execution

```bash
cd StanLogic/tests/KMapSolver
python test_kmapsolver4d_24bit_multicore.py
```

### Configuration

Edit the configuration section in the test file:

```python
NUM_VARS = 24          # Change to 28 or 32 for larger tests
NUM_CORES = 4          # Adjust based on your CPU
RANDOM_SEED = 42       # For reproducibility
```

## Output Files

The test generates several files in `outputs/`:

1. **Core CSV Files**: 
   - `kmapsolver4d_24bit_core0.csv`
   - `kmapsolver4d_24bit_core1.csv`
   - `kmapsolver4d_24bit_core2.csv`
   - `kmapsolver4d_24bit_core3.csv`

2. **Merged CSV**: `kmapsolver4d_24bit_merged.csv`
   - Contains all essential prime implicants from all chunks
   - Format: `chunk_index, extra_combo, bitmask_4x4, term_bits, type`

3. **Results Summary**: `kmapsolver4d_24bit_results.csv`
   - Processing statistics
   - Time measurements
   - Parallel efficiency metrics

## CSV Format

Each row in the CSV represents an essential prime implicant:

| Column | Description |
|--------|-------------|
| `chunk_index` | Which chunk this term belongs to (0-255) |
| `extra_combo` | Binary string for extra variables (12 bits) |
| `bitmask_4x4` | Hexadecimal bitmask covering 4×4 map positions |
| `term_bits` | Bit pattern with '-' for don't-cares (e.g., "01-1") |
| `type` | Always "essential_prime" (essential prime implicant) |

## Distributed Processing

For processing across multiple computers:

### Computer 1 (Cores 0-3):
```python
# Process chunks 0-127
chunk_ranges = [(0, 32), (32, 64), (64, 96), (96, 128)]
```

### Computer 2 (Cores 0-3):
```python
# Process chunks 128-255
chunk_ranges = [(128, 160), (160, 192), (192, 224), (224, 256)]
```

Then merge all CSV files from both computers.

## Scaling to Larger Bit Sizes

### 28-Bit (4,096 chunks)
- **4 cores**: ~21.6 hours
- **8 cores** (2 computers): ~10.8 hours

### 32-Bit (65,536 chunks)
- **4 cores**: ~14.4 days
- **8 cores** (2 computers): ~7.2 days

## Memory Management

The implementation includes aggressive memory management:
- Chunk data deleted immediately after processing
- Explicit garbage collection after each chunk
- Essential prime implicants written directly to CSV (not stored in RAM)
- Each core maintains its own CSV file to avoid locking

## Next Steps

After the test completes:

1. **Verify Results**: Check `kmapsolver4d_24bit_results.csv` for statistics
2. **Merge CSVs**: Already done automatically by the test
3. **Bitwise Union**: Implement final term extraction across all chunks
4. **Analyze Terms**: Study the minimized Boolean expression

## Requirements

- Python 3.8+
- StanLogic library installed
- Multiprocessing support
- Sufficient disk space for CSV files (~100MB for 24-bit)

## Notes

- First 3 chunks of Core 0 show verbose output for monitoring
- Progress updates every 10 chunks per core
- Random seed ensures reproducibility
- CSV files can be deleted after merging to save space
