# Chunked Processing Strategy for 32-bit K-Map Minimization

## Overview

For a 32-bit Boolean function, we need to process 2³² = 4,294,967,296 truth table entries. Creating the full list at once is impossible due to memory constraints. The chunked processing strategy splits this into manageable pieces.

## Simplified Example: 6-bit Function

Let's use a **6-bit function** as a concrete example to understand the strategy. We'll then scale the concept to 32 bits.

### Configuration
- **Total variables**: 6 (labeled A, B, C, D, E, F)
- **K-map variables**: 2 (we'll use E, F for the K-map gray code)
- **Extra variables**: 4 (A, B, C, D are "extra" - they define which chunk we're in)
- **Chunk size**: 2² = 4 entries (all combinations of E, F)
- **Total chunks**: 2⁴ = 16 chunks (all combinations of A, B, C, D)

### Truth Table Structure

The complete 6-bit truth table has 2⁶ = 64 entries:

```
Index | A B C D | E F | Output
------|---------|-----|--------
  0   | 0 0 0 0 | 0 0 |   ?
  1   | 0 0 0 0 | 0 1 |   ?
  2   | 0 0 0 0 | 1 0 |   ?
  3   | 0 0 0 0 | 1 1 |   ?
  4   | 0 0 0 1 | 0 0 |   ?
  5   | 0 0 0 1 | 0 1 |   ?
  ...  | ...     | ... |  ...
 63   | 1 1 1 1 | 1 1 |   ?
```

### Chunking Strategy

We split the 64 entries into **16 chunks of 4 entries each**:

#### **Chunk 0**: A=0, B=0, C=0, D=0 (ABCD = 0000)
```
E F | Output | Minterm
----|--------|--------
0 0 |   1    | m0
0 1 |   1    | m1
1 0 |   0    | -
1 1 |   1    | m3
```
**K-map minimization**: Create a 2-variable K-map with E, F
```
     F=0  F=1
E=0   1    1
E=1   0    1
```
**Result**: E'F' + E'F + EF = E' + EF = **E' + F**

**Bitmask representation**:
- Term 1: `000000|11` (ABCD=0000, covers E'F', E'F → E')
- Term 2: `000000|10` (ABCD=0000, covers E'F, EF → F)

Write to CSV: `[chunk_0, 0b00000011, "term"]`, `[chunk_0, 0b00000010, "term"]`

---

#### **Chunk 1**: A=0, B=0, C=0, D=1 (ABCD = 0001)
```
E F | Output | Minterm
----|--------|--------
0 0 |   0    | -
0 1 |   1    | m5
1 0 |   1    | m6
1 1 |   0    | -
```
**K-map**:
```
     F=0  F=1
E=0   0    1
E=1   1    0
```
**Result**: E'F + EF' (no simplification - XOR pattern)

**Bitmasks**: 
- `000001|01` for E'F (chunk 1, minterm 1)
- `000001|10` for EF' (chunk 1, minterm 2)

Write to CSV: `[chunk_1, 0b00000101, "term"]`, `[chunk_1, 0b00000110, "term"]`

---

#### **Chunk 2**: A=0, B=0, C=1, D=0 (ABCD = 0010)
```
E F | Output | Minterm
----|--------|--------
0 0 |   1    | m8
0 1 |   1    | m9
1 0 |   1    | m10
1 1 |   1    | m11
```
**K-map**:
```
     F=0  F=1
E=0   1    1
E=1   1    1
```
**Result**: All ones → constant 1 for this chunk

**Bitmask**: `000010|1111` (entire chunk is 1)

Write to CSV: `[chunk_2, 0b00001111, "all_ones"]`

---

#### **Chunk 15**: A=1, B=1, C=1, D=1 (ABCD = 1111)
```
E F | Output | Minterm
----|--------|--------
0 0 |   0    | -
0 1 |   0    | -
1 0 |   0    | -
1 1 |   0    | -
```
**Result**: All zeros → no terms needed

Skip writing to CSV (or mark as `[chunk_15, 0, "all_zeros"]`)

---

## Stack Management

The key advantage is **efficient memory usage**:

1. **Generate** chunk 0 outputs (4 values) → **Process** with K-map → **Write** to CSV → **Discard** chunk 0 data
2. **Generate** chunk 1 outputs (4 values) → **Process** with K-map → **Write** to CSV → **Discard** chunk 1 data
3. Continue for all 16 chunks...

At any moment, only **one chunk (4 entries)** is in memory, not all 64!

## Bitwise Union Phase

After all chunks are processed, we read the CSV and perform bitwise union:

**CSV Contents**:
```
chunk_index, bitmask, type
0, 0b00000011, term
0, 0b00000010, term
1, 0b00000101, term
1, 0b00000110, term
2, 0b00001111, all_ones
...
```

**Union operation**: Group similar bitmasks and merge:
- Check if masks differ by only one bit → merge via OR
- Example: `0b00000011` and `0b00000010` differ in bit 0 → can potentially merge

**Final terms**: Extract the minimal set of product terms covering all 1s in the truth table.

---

## Scaling to 32-bit

For the actual **32-bit test**:

- **Total variables**: 32
- **K-map variables**: 4 (variables used for 4-variable K-map minimization)
- **Extra variables**: 28 (define chunk index)
- **Chunk size**: 2⁴ = 16 entries
- **Total chunks**: 2²⁸ = **268,435,456 chunks**

### Example Chunk in 32-bit Context

**Chunk 0**: All 28 high-order bits = 0, vary the 4 low-order bits
```
Bits 31-4: 0000...0000 (28 zeros)
Bits 3-0:  varies (0000 to 1111)

16 truth table entries for this chunk
```

**Chunk 1,000,000**: 
```
Binary representation of 1,000,000 in 28 bits: 00000011110100001001000000
Bits 31-4: 00000011110100001001000000
Bits 3-0:  varies (0000 to 1111)

16 different truth table entries
```

**Chunk 268,435,455** (last chunk):
```
Bits 31-4: 1111...1111 (28 ones)
Bits 3-0:  varies (0000 to 1111)

Final 16 truth table entries
```

## Memory Efficiency

**Without chunking**:
- Need to store: 2³² × 1 byte = 4 GB+ in memory simultaneously
- Impossible for most systems

**With chunking**:
- Active memory: 16 entries × 1 byte = 16 bytes per chunk
- Plus K-map solver overhead: ~few KB
- CSV file grows incrementally on disk
- **Total RAM usage**: < 1 MB at any time!

## Processing Time Estimate

If each chunk takes 0.001 seconds to process:
- Total time: 268,435,456 chunks × 0.001s = 268,435 seconds ≈ **74.5 hours**

With optimizations (constant detection, skipping):
- Actual time may be significantly less

## Visual Summary

```
32-bit Truth Table (4.3 billion entries)
│
├─ Chunk 0: [28 zeros][0000-1111] → 16 entries → Minimize → CSV
│                                      ↓
│                                   Discard chunk 0
│
├─ Chunk 1: [28 zeros + 1][0000-1111] → 16 entries → Minimize → CSV
│                                         ↓
│                                      Discard chunk 1
│
├─ Chunk 2: [28 zeros + 2][0000-1111] → 16 entries → Minimize → CSV
│
│  ... (268 million more chunks) ...
│
└─ Chunk 268,435,455: [28 ones][0000-1111] → 16 entries → Minimize → CSV
                                                ↓
                                            Discard chunk

                        ┌─────────────┐
                        │   CSV File  │
                        │  (all terms)│
                        └──────┬──────┘
                               │
                        Bitwise Union
                               ↓
                        ┌─────────────┐
                        │ Final Terms │
                        │  (minimal)  │
                        └─────────────┘
```

## Key Insights

1. **Sequential Processing**: Each chunk is independent, processed one at a time
2. **Gray Code K-map**: The 4 low-order bits form a standard K-map structure
3. **Fixed Context**: The 28 high-order bits provide context (which chunk)
4. **CSV as Accumulator**: Disk storage accumulates results without RAM pressure
5. **Final Simplification**: Bitwise union merges terms across chunks

This strategy makes the impossible (4GB+ in RAM) possible (KB in RAM, with time tradeoff).
