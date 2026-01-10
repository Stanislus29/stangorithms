```
═══════════════════════════════════════════════════════════════════
ALGORITHM: GEOMETRIC 3D K-MAP MINIMIZATION
Application: Boolean function minimization for 5 ≤ n ≤ 8 variables
═══════════════════════════════════════════════════════════════════

NOTATION AND DEFINITIONS:
    n           - total number of variables
    d = n - 4   - number of depth variables
    m = 2^d     - number of 4×4 K-maps
    δ           - identifier (d-bit binary string)
    ρ           - position in 4×4 K-map (4-bit binary string)
    π           - pattern (bit string with possible '-' don't cares)
    β           - dictionary mapping identifiers to patterns
    Ψ           - set of essential prime implicants
    F           - Boolean function output values
    G           - Gray code sequence ["00", "01", "11", "10"]

DIMENSIONAL STRUCTURE:
    Depth (d bits):    Identifier space (which K-map)
    Breadth (2 bits):  Columns in 4×4 K-map (Gray-coded)
    Length (2 bits):   Rows in 4×4 K-map (Gray-coded)

═══════════════════════════════════════════════════════════════════
PHASE 0: TRUTH TABLE GENERATION & PARTITIONING
═══════════════════════════════════════════════════════════════════

Algorithm: GENERATE-TRUTH-TABLE(n)
Input: n - number of variables
Output: T - truth table (2^n entries)

    T ← []
    current ← [0, 0, ..., 0]  // Start with all zeros
    T.append(current)
    
    for i ← 1 to 2^n - 1 do
        current ← BINARY-ADD-ONE(current)  // Increment binary counter
        T.append(current)
    end for
    
    return T

───────────────────────────────────────────────────────────────────

Function: BINARY-ADD-ONE(bits)
Input: bits - binary number as list
Output: result - bits + 1

    result ← bits.copy()
    carry ← 1
    
    for i ← |bits| - 1 down to 0 do  // Process right to left
        sum ← result[i] + carry
        result[i] ← sum mod 2          // Store bit value
        carry ← sum ÷ 2                // Propagate carry
        
        if carry = 0 then break        // No more carry, done
    end for
    
    return result

═══════════════════════════════════════════════════════════════════
PHASE 1: BUILD HIERARCHICAL K-MAP STRUCTURE
═══════════════════════════════════════════════════════════════════

Algorithm: BUILD-KMAPS(n, T, F)
Input: n - number of variables
       T - truth table
       F - output values for each row
Output: K - dictionary mapping identifiers to 4×4 K-maps

Constants:
    d ← n - 4                          // Number of depth variables
    m ← 2^d                            // Total K-maps needed
    G ← ["00", "01", "11", "10"]      // Gray code ordering
    
    K ← ∅
    
    for i ← 0 to m - 1 do
        δ ← BINARY(i, d)               // Generate identifier
        M ← 4×4 matrix initialized to null
        
        for r ← 0 to 3 do              // For each row
            for c ← 0 to 3 do          // For each column
                σ ← δ ∘ G[c] ∘ G[r]    // Build n-bit string: [depth][col][row]
                idx ← FIND-INDEX(T, σ) // Locate in truth table
                
                M[r][c] ← {
                    minterm: idx,
                    value: F[idx],
                    variables: σ
                }
            end for
        end for
        
        K[δ] ← M                       // Store K-map for this identifier
    end for
    
    return K

───────────────────────────────────────────────────────────────────

Gray Code Mapping:
    Column bits (x_{n-3}, x_{n-2}): 00→0, 01→1, 11→2, 10→3
    Row bits (x_{n-1}, x_n):        00→0, 01→1, 11→2, 10→3

═══════════════════════════════════════════════════════════════════
PHASE 2: SOLVE INDIVIDUAL 4×4 K-MAPS (2D MINIMIZATION)
═══════════════════════════════════════════════════════════════════

Algorithm: SOLVE-SINGLE-KMAP(K[δ], form)
Input: K[δ] - 4×4 K-map for identifier δ
       form - 'sop' or 'pos'
Output: List of 4-bit patterns (with don't cares)

    // Extract value matrix from K-map
    V ← 4×4 matrix
    for r ← 0 to 3 do
        for c ← 0 to 3 do
            V[r][c] ← K[δ][r][c].value
        end for
    end for
    
    // Apply standard 2D K-map minimization
    groups ← FIND-ALL-GROUPS(V, form)      // Find rectangular groups
    primes ← FILTER-PRIME-IMPLICANTS(groups)  // Keep only maximal groups
    
    // Convert groups to bit patterns
    patterns ← []
    for g ∈ primes do
        bits ← EXTRACT-BIT-PATTERN(g, G)   // Encode as 4-bit pattern
        patterns.append(bits)
    end for
    
    // Select minimal covering set
    epis ← SELECT-ESSENTIAL-PIS(patterns, V, form)
    
    return epis

───────────────────────────────────────────────────────────────────

Function: EXTRACT-BIT-PATTERN(group, G)
Input: group - set of cells in K-map
       G - Gray code sequence
Output: 4-bit pattern with don't cares

    cells ← GET-CELLS(group)
    bits ← ['?', '?', '?', '?']
    
    for each position in [col_bit1, col_bit2, row_bit1, row_bit2] do
        values ← {cell.bit[position] : cell ∈ cells}
        
        if |values| = 1 then
            bits[position] ← the single value    // Fixed bit
        else
            bits[position] ← '-'                 // Variable bit (don't care)
        end if
    end for
    
    return bits

═══════════════════════════════════════════════════════════════════
PHASE 3: 3D CLUSTER IDENTIFICATION
═══════════════════════════════════════════════════════════════════

Algorithm: IDENTIFY-3D-CLUSTERS(β, I)
Input: β - dictionary mapping identifier δ → list of patterns
       I - set of all identifiers
Output: Γ - valid 3D clusters

    // Build inverse mapping: which identifiers contain each pattern
    Θ ← ∅
    
    for each δ ∈ I do
        for each π ∈ β[δ] do
            if π ∉ Θ then
                Θ[π] ← []
            end if
            Θ[π].append(δ)
        end for
    end for
    
    // Filter to valid 3D clusters
    Γ ← ∅
    Eliminated ← []
    
    for each (π, identifiers) ∈ Θ do
        if |identifiers| = 1 then
            // Single K-map only: not a 3D cluster
            Eliminated.append((π, identifiers[0]))
            
        else if ¬HAS-ADJACENT-IDENTIFIERS(identifiers) then
            // Non-adjacent identifiers: cannot merge in 3D
            for each δ ∈ identifiers do
                Eliminated.append((π, δ))
            end for
            
        else
            // Valid 3D cluster: spans adjacent K-maps
            Γ[π] ← identifiers
        end if
    end for
    
    // Safety fallback: if no 3D clusters found, use all patterns
    if Γ = ∅ then
        Γ ← Θ
    end if
    
    return Γ

───────────────────────────────────────────────────────────────────

Function: HAS-ADJACENT-IDENTIFIERS(identifiers)
Input: identifiers - list of binary strings
Output: Boolean - true if any pair differs by exactly 1 bit

    // Check for Gray code adjacency (Hamming distance = 1)
    for i ← 0 to |identifiers| - 1 do
        for j ← i+1 to |identifiers| - 1 do
            if HAMMING-DISTANCE(identifiers[i], identifiers[j]) = 1 then
                return TRUE
            end if
        end for
    end for
    return FALSE

───────────────────────────────────────────────────────────────────

Function: HAMMING-DISTANCE(σ₁, σ₂)
Input: σ₁, σ₂ - binary strings
Output: Number of differing bits

    if |σ₁| ≠ |σ₂| then return ∞
    
    count ← 0
    for i ← 0 to |σ₁| - 1 do
        if σ₁[i] ≠ σ₂[i] then
            count ← count + 1
        end if
    end for
    return count

═══════════════════════════════════════════════════════════════════
PHASE 4: DEPTH-WISE MERGING (Quine-McCluskey on Identifiers)
═══════════════════════════════════════════════════════════════════

Algorithm: DEPTH-WISE-MERGE(Γ)
Input: Γ - valid 3D clusters (pattern → identifiers)
Output: Λ - merged 3D clusters with metadata

    Λ ← []
    
    for each (π, identifiers) ∈ Γ do
        // Apply Quine-McCluskey to merge identifiers in depth dimension
        Φ ← QUINE-MCCLUSKEY-COMPLETE(identifiers)
        
        // Create cluster object for each merged identifier pattern
        for each φ ∈ Φ do
            cluster ← {
                kmap_pattern: π,              // 4-bit pattern (breadth×length)
                identifier_pattern: φ,        // d-bit pattern (depth)
                full_pattern: φ ∘ π,          // Complete n-bit pattern
                depth: 2^(COUNT-DASHES(φ)),   // Number of K-maps spanned
                cells: GET-CELL-POSITIONS(π)  // Cell positions in K-map
            }
            Λ.append(cluster)
        end for
    end for
    
    return Λ

═══════════════════════════════════════════════════════════════════
SUBROUTINE: QUINE-MCCLUSKEY WITH ESSENTIAL PI SELECTION
═══════════════════════════════════════════════════════════════════

Algorithm: QUINE-MCCLUSKEY-COMPLETE(M)
Input: M - set of minterms (binary strings)
Output: Ψ - minimal set of prime implicants

    // Handle trivial cases
    if |M| = 0 then return ∅
    if |M| = 1 then return M
    
    M ← REMOVE-DUPLICATES(M)
    
    // Find all prime implicants via iterative merging
    Π ← FIND-ALL-PRIME-IMPLICANTS(M)
    
    // Select minimal covering set
    Ψ ← SELECT-ESSENTIAL-PRIME-IMPLICANTS(Π, M)
    
    return Ψ

───────────────────────────────────────────────────────────────────

Function: FIND-ALL-PRIME-IMPLICANTS(M)
Input: M - set of minterms
Output: Π - set of all prime implicants

    // Convert strings to bitwise representation (value, mask)
    T ← ∅
    for each μ ∈ M do
        (v, m) ← STRING-TO-BITWISE(μ)
        T.add((v, m))
    end for
    
    Π ← ∅
    iteration ← 1
    
    while TRUE do
        T' ← ∅     // Next iteration terms
        U ← ∅      // Terms used in merges
        
        // Try merging all pairs
        for each (v₁, m₁) ∈ T do
            for each (v₂, m₂) ∈ T where (v₂, m₂) > (v₁, m₁) do
                if m₁ ≠ m₂ then continue      // Different don't care positions
                
                diff ← (v₁ ⊕ v₂) ∧ m₁          // Find differing bits
                
                // Check if exactly one bit differs
                if diff ≠ 0 ∧ (diff ∧ (diff - 1)) = 0 then
                    v' ← v₁ ∧ v₂               // Common value bits
                    m' ← m₁ ∧ ¬diff            // Mark differing bit as don't care
                    
                    T'.add((v', m'))
                    U.add((v₁, m₁))            // Mark as used
                    U.add((v₂, m₂))
                end if
            end for
        end for
        
        // Unused terms are prime implicants
        Π ← Π ∪ (T \ U)
        
        if T' = ∅ then break                   // No more merges possible
        T ← T'
        iteration ← iteration + 1
    end while
    
    // Convert back to string representation
    return {BITWISE-TO-STRING(v, m) : (v, m) ∈ Π}

───────────────────────────────────────────────────────────────────

Function: SELECT-ESSENTIAL-PRIME-IMPLICANTS(Π, M)
Input: Π - set of prime implicants
       M - set of minterms
Output: Ψ - minimal set of essential prime implicants

    // Build coverage table: which minterms each PI covers
    Cov ← ∅
    for each π ∈ Π do
        Cov[π] ← {μ ∈ M : IMPLICANT-COVERS-MINTERM(π, μ)}
    end for
    
    Ψ ← []     // Selected essential PIs
    V ← ∅      // Covered minterms
    U ← M      // Uncovered minterms
    
    // Step 1: Select essential prime implicants
    for each μ ∈ M do
        covering ← {π ∈ Π : μ ∈ Cov[π]}
        
        if |covering| = 1 then
            // This minterm is covered by only one PI → essential
            π ← the single element in covering
            if π ∉ Ψ then
                Ψ.append(π)
                V ← V ∪ Cov[π]
                U ← U \ Cov[π]
            end if
        end if
    end for
    
    // Step 2: Greedy cover for remaining uncovered minterms
    Π' ← Π \ Ψ
    
    while U ≠ ∅ ∧ Π' ≠ ∅ do
        // Select PI covering most uncovered minterms
        π* ← arg max_{π ∈ Π'} |Cov[π] ∩ U|
        
        if |Cov[π*] ∩ U| = 0 then break
        
        Ψ.append(π*)
        newly_covered ← Cov[π*] ∩ U
        V ← V ∪ newly_covered
        U ← U \ newly_covered
        Π' ← Π' \ {π*}
    end while
    
    return Ψ

───────────────────────────────────────────────────────────────────

Function: IMPLICANT-COVERS-MINTERM(π, μ)
Input: π - pattern with possible don't cares
       μ - concrete minterm
Output: Boolean - true if π covers μ

    if |π| ≠ |μ| then return FALSE
    
    for i ← 0 to |π| - 1 do
        if π[i] ≠ '-' ∧ π[i] ≠ μ[i] then
            return FALSE
        end if
    end for
    
    return TRUE

═══════════════════════════════════════════════════════════════════
PHASE 5: ESSENTIAL PRIME IMPLICANT SELECTION (Depth Dominance)
═══════════════════════════════════════════════════════════════════

Algorithm: SELECT-EPIS-BY-DEPTH-DOMINANCE(Λ)
Input: Λ - merged 3D clusters
Output: Ψ - essential prime implicants

    // Group clusters by their K-map pattern
    G ← ∅
    for each λ ∈ Λ do
        π ← λ.kmap_pattern
        if π ∉ G then G[π] ← []
        G[π].append(λ)
    end for
    
    Ψ ← []
    
    for each (π, clusters) ∈ G do
        // Further group by cell coverage (which cells they cover)
        H ← ∅
        for each λ ∈ clusters do
            ω ← λ.cells
            if ω ∉ H then H[ω] ← []
            H[ω].append(λ)
        end for
        
        // For each cell group, keep only maximum depth clusters
        for each (ω, cell_group) ∈ H do
            d_max ← max{λ.depth : λ ∈ cell_group}
            
            // Select clusters with maximum depth (they dominate others)
            for each λ ∈ cell_group do
                if λ.depth = d_max then
                    Ψ.append(λ)  // Maximum depth → essential
                end if
            end for
        end for
    end for
    
    return Ψ

───────────────────────────────────────────────────────────────────

Dominance Criterion:
    λ₁ dominates λ₂ ⟺ 
        (λ₁.kmap_pattern = λ₂.kmap_pattern) ∧
        (λ₁.cells = λ₂.cells) ∧
        (λ₁.depth > λ₂.depth)

───────────────────────────────────────────────────────────────────

Function: GET-CELL-POSITIONS(π)
Input: π - 4-bit pattern with possible don't cares
Output: Set of (row, col) positions

    G ← ['00', '01', '11', '10']
    concrete ← EXPAND-PATTERN(π)       // Expand don't cares to all possibilities
    
    cells ← ∅
    for each σ ∈ concrete do
        col_bits ← σ[0:2]              // First 2 bits = column
        row_bits ← σ[2:4]              // Last 2 bits = row
        
        c ← GRAY-INDEX(col_bits, G)    // Map Gray code to index
        r ← GRAY-INDEX(row_bits, G)
        
        cells.add((r, c))
    end for
    
    return frozenset(cells)

───────────────────────────────────────────────────────────────────

Function: EXPAND-PATTERN(π)
Input: π - pattern with don't cares
Output: Set of all concrete strings matching π

    n_dashes ← COUNT-DASHES(π)
    
    if n_dashes = 0 then return {π}    // No don't cares, return as-is
    
    // Generate all 2^n_dashes combinations
    result ← ∅
    for i ← 0 to 2^n_dashes - 1 do
        σ ← π.copy()
        bits ← BINARY(i, n_dashes)     // Binary representation of i
        bit_idx ← 0
        
        // Replace each '-' with corresponding bit from i
        for j ← 0 to |σ| - 1 do
            if σ[j] = '-' then
                σ[j] ← bits[bit_idx]
                bit_idx ← bit_idx + 1
            end if
        end for
        
        result.add(σ)
    end for
    
    return result

═══════════════════════════════════════════════════════════════════
PHASE 6: COVERAGE VERIFICATION & 2D PATTERN COLLECTION
═══════════════════════════════════════════════════════════════════

Algorithm: VERIFY-AND-COLLECT-2D(Ψ, β, K, I, M_all)
Input: Ψ - selected 3D EPIs
       β - dictionary mapping identifier → patterns
       K - all K-maps
       I - all identifiers
       M_all - all target minterms
Output: Ψ_complete - complete coverage with 2D patterns

    // Get patterns from 3D EPIs
    Π₃ᴰ ← {λ.full_pattern : λ ∈ Ψ}
    
    // Check which minterms are covered by 3D EPIs
    M_cov₃ᴰ ← ⋃_{π ∈ Π₃ᴰ} EXPAND-PATTERN(π)
    
    // Find uncovered minterms
    M_unc ← M_all \ M_cov₃ᴰ
    
    if M_unc ≠ ∅ then
        // Collect 2D patterns for coverage
        Π₂ᴰ ← COLLECT-2D-FOR-COVERAGE(β, Π₃ᴰ, M_unc, M_all)
        
        // Combine 3D and 2D patterns
        Π_all ← Π₃ᴰ ∪ Π₂ᴰ
        
        // Verify complete coverage
        M_cov_final ← ⋃_{π ∈ Π_all} EXPAND-PATTERN(π)
        M_still_unc ← M_all \ M_cov_final
        
        if M_still_unc ≠ ∅ then
            // Apply fallback coverage
            Π_fallback ← FALLBACK-COVERAGE(M_still_unc)
            Π_all ← Π_all ∪ Π_fallback
        end if
    else
        Π_all ← Π₃ᴰ
    end if
    
    return Π_all

───────────────────────────────────────────────────────────────────

Function: COLLECT-2D-FOR-COVERAGE(β, Π₃ᴰ, M_unc, M_all)
Input: β - dictionary mapping identifier → patterns
       Π₃ᴰ - set of 3D patterns already selected
       M_unc - uncovered minterms
       M_all - all target minterms
Output: Π₂ᴰ - set of 2D patterns for coverage

    if M_unc = ∅ then
        return ∅                         // All covered by 3D patterns
    end if
    
    // Build list of 2D candidate patterns
    candidates ← []
    
    for each δ ∈ β do
        for each π ∈ β[δ] do
            π_full ← δ ∘ π               // Full pattern
            
            if π_full ∈ Π₃ᴰ then
                continue                 // Skip 3D patterns
            end if
            
            // Calculate coverage
            M_covered ← EXPAND-PATTERN(π_full) ∩ M_all
            M_overlap ← M_covered ∩ M_unc
            
            if M_overlap ≠ ∅ then
                candidates.append({
                    pattern: π_full,
                    covered: M_covered,
                    unc_count: |M_overlap|
                })
            end if
        end for
    end for
    
    // Find essential 2D patterns
    essential ← []
    M_cov_essential ← ∅
    
    // Build minterm-to-patterns mapping
    MT_map ← ∅
    for each c ∈ candidates do
        for each m ∈ (c.covered ∩ M_unc) do
            MT_map[m].append(c)
        end for
    end for
    
    // Select essential patterns (only pattern covering a minterm)
    for each (m, patterns) ∈ MT_map do
        if |patterns| = 1 then
            c ← patterns[0]
            if c ∉ essential then
                essential.append(c)
                M_cov_essential ← M_cov_essential ∪ c.covered
            end if
        end if
    end for
    
    // Greedy set cover for remaining
    selected ← essential
    M_remaining ← M_unc \ M_cov_essential
    available ← candidates \ essential
    
    while M_remaining ≠ ∅ ∧ available ≠ ∅ do
        // Select pattern covering most remaining minterms
        best ← arg max_{c ∈ available} |c.covered ∩ M_remaining|
        
        if |best.covered ∩ M_remaining| = 0 then
            break
        end if
        
        selected.append(best)
        M_remaining ← M_remaining \ best.covered
        available.remove(best)
    end while
    
    return {c.pattern : c ∈ selected}

───────────────────────────────────────────────────────────────────

Function: FALLBACK-COVERAGE(M_unc)
Input: M_unc - uncovered minterms
Output: Set of fallback patterns (direct minterm patterns)

    // Direct coverage: use each minterm as its own pattern
    return M_unc

───────────────────────────────────────────────────────────────────

Function: GET-ALL-MINTERMS(K, I)
Input: K - all K-maps
       I - all identifiers
Output: Set of all minterms where F = 1

    M ← ∅
    
    for each δ ∈ I do
        for r ← 0 to 3 do
            for c ← 0 to 3 do
                cell ← K[δ][r][c]
                if cell.value = 1 then
                    σ ← δ ∘ CELL-TO-BITS(r, c)
                    M.add(σ)
                end if
            end for
        end for
    end for
    
    return M



═══════════════════════════════════════════════════════════════════
PHASE 7: FINAL QUINE-MCCLUSKEY OPTIMIZATION
═══════════════════════════════════════════════════════════════════

Algorithm: OPTIMIZE-WITH-QUINE-MCCLUSKEY(Π_all, M_all)
Input: Π_all - set of all patterns (3D + 2D + fallback)
       M_all - all target minterms
Output: Π_optimal - globally minimal set of patterns

    if |Π_all| ≤ 1 then
        return Π_all
    end if
    
    // Expand patterns to verify coverage
    M_covered ← ⋃_{π ∈ Π_all} EXPAND-PATTERN(π)
    
    if M_covered ≠ M_all then
        // Coverage mismatch - return patterns as-is
        return Π_all
    end if
    
    // Re-minimize from scratch using complete Quine-McCluskey
    Π_prime ← FIND-ALL-PRIME-IMPLICANTS(M_all)
    
    // Filter out invalid prime implicants
    Π_valid ← ∅
    for each π ∈ Π_prime do
        if EXPAND-PATTERN(π) ⊆ M_all then
            Π_valid ← Π_valid ∪ {π}
        end if
    end for
    
    // Select essential PIs for minimal cover
    Π_optimal ← SELECT-ESSENTIAL-PRIME-IMPLICANTS(Π_valid, M_all)
    
    return Π_optimal

───────────────────────────────────────────────────────────────────

Note: This final optimization step ensures global minimality by:
    1. Expanding all patterns back to their minterms
    2. Re-applying Quine-McCluskey from scratch on the minterm set
    3. Filtering invalid prime implicants that cover unwanted minterms
    4. Selecting essential prime implicants for minimal cover
    
This allows patterns from different sources (3D, 2D, fallback) to be
merged and optimized together, producing a globally minimal result.

═══════════════════════════════════════════════════════════════════
PHASE 8: EXPRESSION GENERATION
═══════════════════════════════════════════════════════════════════

Algorithm: GENERATE-EXPRESSION(Ψ, form, n)
Input: Ψ - essential prime implicants
       form - 'sop' or 'pos'
       n - number of variables
Output: (T, E) - list of terms and expression string

    T ← []
    
    for each λ ∈ Ψ do
        τ ← PATTERN-TO-TERM(λ.full_pattern, form, n)
        T.append(τ)
    end for
    
    if form = 'sop' then
        E ← JOIN(T, " + ")
    else if form = 'pos' then
        E ← JOIN(T, " · ")
    end if
    
    return (T, E)

───────────────────────────────────────────────────────────────────

Function: PATTERN-TO-TERM(π, form, n)
Input: π - bit pattern with don't cares
       form - 'sop' or 'pos'
       n - number of variables
Output: Term string

    vars ← [x₁, x₂, ..., xₙ]
    
    if form = 'sop' then
        literals ← []
        for i ← 0 to |π| - 1 do
            if π[i] = '0' then
                literals.append(vars[i] + "'")
            else if π[i] = '1' then
                literals.append(vars[i])
            end if
        end for
        
        return CONCATENATE(literals)
        
    else if form = 'pos' then
        literals ← []
        for i ← 0 to |π| - 1 do
            if π[i] = '1' then
                literals.append(vars[i] + "'")
            else if π[i] = '0' then
                literals.append(vars[i])
            end if
        end for
        
        return "(" + JOIN(literals, " + ") + ")"
    end if

═══════════════════════════════════════════════════════════════════
MAIN ALGORITHM: 3D K-MAP MINIMIZATION
═══════════════════════════════════════════════════════════════════

Algorithm: MINIMIZE-3D(n, F, form)
Input: n - number of variables (5 ≤ n ≤ 8)
       F - output values (2^n entries)
       form - 'sop' or 'pos'
Output: (T, E) - minimized terms and expression

Preconditions:
    5 ≤ n ≤ 8
    |F| = 2^n

Begin:
    T ← GENERATE-TRUTH-TABLE(n)
    K ← BUILD-KMAPS(n, T, F)
    I ← KEYS(K)
    
    // Get all target minterms
    M_all ← GET-ALL-MINTERMS(K, I)
    
    // Build pattern dictionary from 2D minimization
    β ← ∅
    for each δ ∈ SORTED(I) do
        patterns ← SOLVE-SINGLE-KMAP(K[δ], form)
        β[δ] ← patterns
    end for
    
    // 3D clustering and merging
    Γ ← IDENTIFY-3D-CLUSTERS(β, I)
    Λ ← DEPTH-WISE-MERGE(Γ)
    Ψ ← SELECT-EPIS-BY-DEPTH-DOMINANCE(Λ)
    
    // Coverage verification and 2D pattern collection
    Π_all ← VERIFY-AND-COLLECT-2D(Ψ, β, K, I, M_all)
    
    // Final global optimization
    Π_optimal ← OPTIMIZE-WITH-QUINE-MCCLUSKEY(Π_all, M_all)
    
    // Generate expression
    (T, E) ← GENERATE-EXPRESSION(Π_optimal, form, n)
    
    return (T, E)

End Algorithm

═══════════════════════════════════════════════════════════════════
END OF ALGORITHM
═══════════════════════════════════════════════════════════════════
```