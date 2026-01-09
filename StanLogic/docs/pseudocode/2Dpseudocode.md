```
═══════════════════════════════════════════════════════════════════
ALGORITHM: 2D K-MAP MINIMIZATION (Clustering with Bitmask Optimization)
Application: Boolean function minimization for 2-4 variables
═══════════════════════════════════════════════════════════════════

NOTATION AND DEFINITIONS:
    K           - K-map matrix (2D array)
    n           - number of variables (2, 3, or 4)
    R           - number of rows in K-map
    C           - number of columns in K-map
    G           - Gray code sequence ["00", "01", "11", "10"]
    τ           - target value (1 for SOP, 0 for POS)
    μ           - bitmask (integer where bit i represents cell i)
    Π           - set of prime implicants
    Ψ           - set of essential prime implicants
    ω           - Boolean term/expression

DIMENSIONAL STRUCTURE:
    2-var: 2×2 K-map (4 cells)
    3-var: 2×4 or 4×2 K-map (8 cells)  
    4-var: 4×4 K-map (16 cells)

VARIABLE CONVENTIONS:
    Vranesic (default):  cols = x₁x₂, rows = x₃x₄
    Mano-Kime:          rows = x₁x₂, cols = x₃x₄

═══════════════════════════════════════════════════════════════════
PHASE 0: INITIALIZATION & PREPROCESSING
═══════════════════════════════════════════════════════════════════

Algorithm: INITIALIZE-KMAP(K, convention)
Input: K - 2D K-map matrix
       convention - variable ordering ("vranesic" or "mano_kime")
Output: Preprocessed K-map with lookup tables

    R ← |K|                              // Number of rows
    C ← |K[0]|                           // Number of columns
    size ← R × C
    
    // Determine number of variables and Gray code labels
    if size = 4 then
        n ← 2
        row_labels ← ["0", "1"]
        col_labels ← ["0", "1"]
        
    else if size = 8 then
        n ← 3
        if R = 2 ∧ C = 4 then
            row_labels ← ["0", "1"]
            col_labels ← ["00", "01", "11", "10"]  // Gray code
        else if R = 4 ∧ C = 2 then
            row_labels ← ["00", "01", "11", "10"]  // Gray code
            col_labels ← ["0", "1"]
        end if
        
    else if size = 16 then
        n ← 4
        row_labels ← ["00", "01", "11", "10"]  // Gray code
        col_labels ← ["00", "01", "11", "10"]  // Gray code
        
    else
        ERROR: "Invalid K-map size"
    end if
    
    // Define valid rectangular group sizes (powers of 2)
    // Format: (height, width)
    S ← {(1,1), (1,2), (2,1), (2,2), (1,4), (4,1), (2,4), (4,2), (4,4)}
    
    // Filter to dimensions that fit this K-map
    S ← {(h,w) ∈ S : h ≤ R ∧ w ≤ C}
    
    // Precompute lookup tables for efficient access
    // cell_index: (row, col) → minterm index
    // cell_bits: (row, col) → binary string
    for r ← 0 to R-1 do
        for c ← 0 to C-1 do
            cell_index[r,c] ← CELL-TO-TERM(r, c, row_labels, col_labels)
            cell_bits[r,c] ← CELL-TO-BITS(r, c, row_labels, col_labels, convention)
        end for
    end for
    
    // Reverse lookup: term index → (row, col)
    for r ← 0 to R-1 do
        for c ← 0 to C-1 do
            index_to_rc[cell_index[r,c]] ← (r, c)
        end for
    end for
    
    return (K, n, R, C, S, cell_index, cell_bits, index_to_rc)

───────────────────────────────────────────────────────────────────

Function: CELL-TO-TERM(r, c, row_labels, col_labels)
Input: r, c - cell coordinates
       row_labels, col_labels - Gray code labels
Output: Minterm/maxterm index

    // Convert cell coordinates to minterm/maxterm index
    bits ← col_labels[c] ∘ row_labels[r]  // Concatenation
    return BINARY-TO-INT(bits)

───────────────────────────────────────────────────────────────────

Function: CELL-TO-BITS(r, c, row_labels, col_labels, convention)
Input: r, c - cell coordinates
       row_labels, col_labels - Gray code labels
       convention - variable ordering
Output: Binary string representation

    // Get binary representation based on convention
    if convention = "mano_kime" then
        bits ← row_labels[r] ∘ col_labels[c]
    else  // vranesic (default)
        bits ← col_labels[c] ∘ row_labels[r]
    end if
    return bits

═══════════════════════════════════════════════════════════════════
PHASE 1: RECTANGULAR GROUP ENUMERATION (CLUSTERING)
═══════════════════════════════════════════════════════════════════

Algorithm: FIND-ALL-GROUPS(K, τ, allow_dontcare)
Input: K - K-map matrix
       τ - target value (1 for SOP, 0 for POS)
       allow_dontcare - boolean flag for don't care inclusion
Output: Γ - set of group bitmasks

    Γ ← ∅  // Set of valid groups (represented as bitmasks)
    
    // Enumerate all possible rectangular groups
    // Try all starting positions and valid sizes
    for r ← 0 to R-1 do
        for c ← 0 to C-1 do
            for each (h, w) ∈ S do
                // Get cell coordinates with toroidal wrapping
                coords ← GET-GROUP-COORDS(r, c, h, w, R, C)
                
                // Validate group contents
                if allow_dontcare then
                    // Must contain at least one τ, and only τ or 'd'
                    valid ← (∃ (rᵢ, cᵢ) ∈ coords : K[rᵢ,cᵢ] = τ) ∧
                            (∀ (rᵢ, cᵢ) ∈ coords : K[rᵢ,cᵢ] ∈ {τ, 'd'})
                else
                    // Must contain only τ
                    valid ← ∀ (rᵢ, cᵢ) ∈ coords : K[rᵢ,cᵢ] = τ
                end if
                
                if valid then
                    // Create bitmask representation
                    μ ← 0
                    for each (rᵢ, cᵢ) ∈ coords do
                        μ ← μ | (1 << cell_index[rᵢ,cᵢ])
                    end for
                    Γ ← Γ ∪ {μ}
                end if
            end for
        end for
    end for
    
    return Γ

───────────────────────────────────────────────────────────────────

Function: GET-GROUP-COORDS(r, c, h, w, R, C)
Input: r, c - starting position
       h, w - group dimensions
       R, C - K-map dimensions
Output: Set of cell coordinates

    // Generate cell coordinates with toroidal K-map wrapping
    // Adjacent edges are considered connected
    coords ← ∅
    for i ← 0 to h-1 do
        for j ← 0 to w-1 do
            rᵢ ← (r + i) mod R         // Row wrapping
            cᵢ ← (c + j) mod C         // Column wrapping
            coords ← coords ∪ {(rᵢ, cᵢ)}
        end for
    end for
    return coords

───────────────────────────────────────────────────────────────────

Wrapping Rules (Torus Topology):
    Top edge connects to bottom edge
    Left edge connects to right edge
    Corner cells have 4 neighbors
    Edge cells (non-corner) have 3 neighbors

Group Size Constraints:
    Must be power of 2: 1, 2, 4, 8, 16
    Must be rectangular
    Can wrap around edges

═══════════════════════════════════════════════════════════════════
PHASE 2: PRIME IMPLICANT IDENTIFICATION (MAXIMAL CLUSTERS)
═══════════════════════════════════════════════════════════════════

Algorithm: FILTER-PRIME-IMPLICANTS(Γ)
Input: Γ - set of all valid groups (bitmasks)
Output: Π - set of prime implicants (maximal groups)

    Π ← []
    
    // Sort by size (number of cells) descending for efficiency
    Γ_sorted ← SORT(Γ, key=POPCOUNT, reverse=TRUE)
    
    for each μ ∈ Γ_sorted do
        is_subset ← FALSE
        
        // Check if μ is a proper subset of any other group
        for each ν ∈ Γ_sorted do
            if ν = μ then continue
            
            // Bitwise subset check: μ ⊆ ν ⟺ (μ ∧ ν) = μ
            if (μ ∧ ν) = μ then
                is_subset ← TRUE     // μ is subsumed by ν
                break
            end if
        end for
        
        if ¬is_subset then
            Π ← Π ∪ {μ}              // μ is maximal (prime implicant)
        end if
    end for
    
    return Π

───────────────────────────────────────────────────────────────────

Prime Implicant Definition:
    μ is a prime implicant ⟺ 
        (μ covers some target cells) ∧
        (∄ν ∈ Γ : ν ≠ μ ∧ μ ⊂ ν)

Bitmask Operations:
    μ ∧ ν:        Bitwise AND (intersection)
    μ | ν:        Bitwise OR (union)
    μ ⊕ ν:        Bitwise XOR (symmetric difference)
    POPCOUNT(μ):  Number of 1-bits (cells in group)
    μ << k:       Left shift by k positions
    μ & -μ:       Isolate lowest set bit

═══════════════════════════════════════════════════════════════════
PHASE 3: TERM SIMPLIFICATION (BIT PATTERN ANALYSIS)
═══════════════════════════════════════════════════════════════════

Algorithm: SIMPLIFY-GROUP-TO-TERM(μ, form)
Input: μ - group bitmask
       form - 'sop' or 'pos'
Output: ω - Boolean term string

    // Extract cell coordinates from bitmask
    coords ← ∅
    bits_list ← []
    
    temp ← μ
    while temp ≠ 0 do
        // Isolate lowest set bit
        low ← temp ∧ (-temp)
        idx ← LOG₂(low)              // Bit position
        temp ← temp - low
        
        (r, c) ← index_to_rc[idx]
        coords ← coords ∪ {(r, c)}
        bits_list ← bits_list ∪ {cell_bits[r,c]}
    end while
    
    // Compare all bit positions to identify don't cares
    simplified ← LIST(bits_list[0])  // Copy first bit string
    
    for each b ∈ bits_list[1:] do
        for i ← 0 to n-1 do
            if simplified[i] ≠ b[i] then
                simplified[i] ← '-'  // Varying position → don't care
            end if
        end for
    end for
    
    // Convert bit pattern to Boolean term
    vars ← [x₁, x₂, ..., xₙ]
    
    if form = 'sop' then
        // Sum of Products term (minterm)
        literals ← []
        for i ← 0 to n-1 do
            if simplified[i] = '0' then
                literals ← literals ∪ {vars[i] + "'"}  // Complement
            else if simplified[i] = '1' then
                literals ← literals ∪ {vars[i]}        // Uncomplemented
            // else '-': don't care, omit variable
        end for
        ω ← CONCATENATE(literals)
        
    else if form = 'pos' then
        // Product of Sums term (maxterm)
        literals ← []
        for i ← 0 to n-1 do
            if simplified[i] = '1' then
                literals ← literals ∪ {vars[i] + "'"}  // Complement
            else if simplified[i] = '0' then
                literals ← literals ∪ {vars[i]}        // Uncomplemented
            // else '-': don't care, omit variable
        end for
        ω ← "(" + JOIN(literals, " + ") + ")"
    end if
    
    return ω

───────────────────────────────────────────────────────────────────

Bit Simplification Rules:
    Position has '-' (don't care) ⟺ values differ across group
    Variable eliminated from term ⟺ position is '-'

═══════════════════════════════════════════════════════════════════
PHASE 4: ESSENTIAL PRIME IMPLICANT SELECTION
═══════════════════════════════════════════════════════════════════

Algorithm: FIND-ESSENTIAL-PRIME-IMPLICANTS(Π, K, τ)
Input: Π - prime implicants (bitmasks)
       K - K-map matrix
       τ - target value
Output: E - essential prime implicants (indices)
        C - coverage bitmasks for each prime

    // Compute coverage for each prime implicant
    C ← []
    
    for each μ ∈ Π do
        cov ← 0                      // Coverage mask for this prime
        
        temp ← μ
        while temp ≠ 0 do
            low ← temp ∧ (-temp)
            idx ← LOG₂(low)
            temp ← temp - low
            
            (r, c) ← index_to_rc[idx]
            
            // Include in coverage only if cell has target value
            if K[r,c] = τ then
                cov ← cov | (1 << idx)
            end if
        end while
        
        C ← C ∪ {cov}
    end for
    
    // Build minterm-to-primes mapping
    M ← ∅                            // minterm_index → list of prime indices
    all_minterms ← 0
    
    for p ← 0 to |Π|-1 do
        all_minterms ← all_minterms | C[p]
        
        temp ← C[p]
        while temp ≠ 0 do
            low ← temp ∧ (-temp)
            idx ← LOG₂(low)
            temp ← temp - low
            
            M[idx] ← M[idx] ∪ {p}
        end while
    end for
    
    // Identify essential prime implicants
    E ← ∅
    
    for each (m, primes) ∈ M do
        if |primes| = 1 then
            // This minterm is covered by only one prime → essential
            p ← primes[0]
            E ← E ∪ {p}
        end if
    end for
    
    return E, C

───────────────────────────────────────────────────────────────────

Essential Prime Implicant Definition:
    Prime π is essential ⟺ 
        ∃ minterm m : (m is covered only by π)

Coverage Invariant:
    ∀ minterm m with K[m] = τ : 
        ∃ at least one prime π : m is covered by π

═══════════════════════════════════════════════════════════════════
PHASE 5: GREEDY SET COVER
═══════════════════════════════════════════════════════════════════

Algorithm: GREEDY-SET-COVER(Π, E, C, all_minterms)
Input: Π - all prime implicants
       E - essential prime implicants (indices)
       C - coverage masks
       all_minterms - bitmask of all target minterms
Output: S - selected prime implicants

    // Track coverage by essential primes
    covered ← 0
    for each e ∈ E do
        covered ← covered | C[e]
    end for
    
    // Initialize selection with essentials
    S ← E
    
    // Find uncovered minterms
    remaining ← all_minterms ∧ (¬covered)
    
    // Greedy selection loop
    while remaining ≠ 0 do
        // Find prime covering most uncovered minterms
        best_idx ← -1
        best_count ← -1
        
        for p ← 0 to |Π|-1 do
            if p ∈ S then continue   // Already selected
            
            // Count newly covered minterms
            new_coverage ← C[p] ∧ remaining
            count ← POPCOUNT(new_coverage)
            
            if count > best_count then
                best_count ← count
                best_idx ← p
            end if
        end for
        
        // Termination check
        if best_idx = -1 ∨ best_count = 0 then
            break
        end if
        
        // Add best prime to selection
        S ← S ∪ {best_idx}
        covered ← covered | C[best_idx]
        remaining ← all_minterms ∧ (¬covered)
    end while
    
    return S

───────────────────────────────────────────────────────────────────

Greedy Heuristic:
    At each step, select prime implicant that covers
    the maximum number of uncovered minterms

Time Complexity: O(|Π|² × m) where m = number of minterms
Approximation: O(log m) approximation to optimal set cover

═══════════════════════════════════════════════════════════════════
PHASE 6: REDUNDANCY ELIMINATION
═══════════════════════════════════════════════════════════════════

Algorithm: REMOVE-REDUNDANCY(S, C, all_minterms)
Input: S - selected prime implicants
       C - coverage masks
       all_minterms - target minterms
Output: S' - minimal selection without redundancy

    Function: COVERAGE(indices)
    Input: indices - set of prime indices
    Output: Combined coverage mask
    
        // Compute combined coverage for given indices
        mask ← 0
        for each i ∈ indices do
            mask ← mask | C[i]
        end for
        return mask
    End Function
    
    S' ← S                           // Working copy
    target_coverage ← COVERAGE(S)
    
    // Try removing each prime implicant
    for each idx ∈ SORTED(S') do
        trial ← S' \ {idx}           // Try without this prime
        
        if COVERAGE(trial) = target_coverage then
            // Removal maintains full coverage → redundant
            S' ← trial
        end if
    end for
    
    return S'

───────────────────────────────────────────────────────────────────

Redundancy Check:
    Prime π is redundant ⟺ 
        (S \ {π}) covers all minterms covered by S

Post-condition:
    No prime in S' can be removed without losing coverage

═══════════════════════════════════════════════════════════════════
PHASE 7: EXPRESSION ASSEMBLY
═══════════════════════════════════════════════════════════════════

Algorithm: GENERATE-EXPRESSION(S, Π, form)
Input: S - final selected primes (indices)
       Π - all prime implicants (bitmasks)
       form - 'sop' or 'pos'
Output: (T, E) - list of terms and complete expression

    T ← []
    
    // Generate term for each selected prime
    for each idx ∈ SORTED(S) do
        μ ← Π[idx]
        ω ← SIMPLIFY-GROUP-TO-TERM(μ, form)
        T ← T ∪ {ω}
    end for
    
    // Join terms with appropriate operator
    if form = 'sop' then
        E ← JOIN(T, " + ")           // Sum of products
        if T = ∅ then E ← "0"
    else if form = 'pos' then
        E ← JOIN(T, " · ")           // Product of sums
        if T = ∅ then E ← "1"
    end if
    
    return (T, E)

───────────────────────────────────────────────────────────────────

Expression Forms:
    SOP (Sum of Products):    F = t₁ + t₂ + ... + tₙ
    POS (Product of Sums):    F = (s₁)(s₂)...(sₙ)

Empty Expression:
    SOP: F = 0  (no minterms)
    POS: F = 1  (no maxterms)

═══════════════════════════════════════════════════════════════════
MAIN ALGORITHM: 2D K-MAP MINIMIZATION
═══════════════════════════════════════════════════════════════════

Algorithm: MINIMIZE-KMAP-2D(K, form, convention)
Input: K - K-map matrix (values: 0, 1, or 'd')
       form - 'sop' or 'pos'
       convention - variable ordering
Output: (T, E) - minimized terms and expression

Preconditions:
    2 ≤ n ≤ 4
    K is rectangular with power-of-2 size
    form ∈ {'sop', 'pos'}

Begin:
    // PHASE 0: Initialize K-map structure
    (K, n, R, C, S, cell_index, cell_bits, index_to_rc) ← 
        INITIALIZE-KMAP(K, convention)
    
    // Set target value based on form
    τ ← 0 if form = 'pos' else 1
    
    // PHASE 1: Enumerate all valid rectangular groups
    Γ ← FIND-ALL-GROUPS(K, τ, allow_dontcare=TRUE)
    
    // PHASE 2: Filter to maximal groups (prime implicants)
    Π ← FILTER-PRIME-IMPLICANTS(Γ)
    
    // Handle trivial case: no prime implicants
    if Π = ∅ then
        if form = 'sop' then
            return ([], "0")
        else
            return ([], "1")
        end if
    end if
    
    // PHASE 3: Identify essential prime implicants
    (E, C) ← FIND-ESSENTIAL-PRIME-IMPLICANTS(Π, K, τ)
    
    // Compute total minterms to cover
    all_minterms ← REDUCE(C, OPERATOR=OR)
    
    // PHASE 4: Greedy set cover for remaining minterms
    S ← GREEDY-SET-COVER(Π, E, C, all_minterms)
    
    // PHASE 5: Remove redundant primes
    S ← REMOVE-REDUNDANCY(S, C, all_minterms)
    
    // PHASE 6: Generate final Boolean expression
    (T, E) ← GENERATE-EXPRESSION(S, Π, form)
    
    return (T, E)

End Algorithm

═══════════════════════════════════════════════════════════════════
END OF ALGORITHM
═══════════════════════════════════════════════════════════════════
```