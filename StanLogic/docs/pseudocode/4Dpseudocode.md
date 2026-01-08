```
═══════════════════════════════════════════════════════════════════
ALGORITHM: GEOMETRIC 4D K-MAP MINIMIZATION
Application: Boolean function minimization for 9 ≤ n ≤ 10  variables
═══════════════════════════════════════════════════════════════════

Algorithm: 4D-MINIMIZE(K, n, form)
Input: K - n-variable Karnaugh map structure (n > 8)
       n - number of variables
       form - minimization form ∈ {SOP, POS}
Output: Ψ - minimal set of prime implicants
        E - Boolean expression string

Constants:
    c ← n - 8                    // Chunk dimension bits (span variables)
    m ← 8                        // Variables per chunk (fixed 8-var blocks)
    
Dimensional Structure:
    Span (c bits):     Chunk identifiers (4th dimension)
    Depth (4 bits):    3D identifiers within each chunk
    Breadth (2 bits):  Columns in 2D K-map
    Length (2 bits):   Rows in 2D K-map

═══════════════════════════════════════════════════════════════════
PHASE 1: CHUNK PARTITIONING
═══════════════════════════════════════════════════════════════════

Function: PARTITION-INTO-CHUNKS(K, n)
Input: K - full K-map structure
       n - total number of variables
Output: C - dictionary mapping chunk identifiers to 8-var K-maps

    c ← n - 8                            // Number of chunk identifier bits
    C ← ∅
    
    for i ← 0 to 2^c - 1 do
        χᵢ ← BINARY(i, c)                // Generate chunk identifier (c bits)
        K^χᵢ ← ∅                         // Initialize 8-variable K-map for this chunk
        
        for j ← 0 to 2^4 - 1 do
            δⱼ ← BINARY(j, 4)             // Inner 3D identifier (4 bits)
            ι ← χᵢ ∘ δⱼ                   // Full identifier: [chunk][depth]
            
            if ι ∈ K then
                K^χᵢ[δⱼ] ← K[ι]           // Extract 4×4 K-map for this position
            end if
        end for
        
        C[χᵢ] ← K^χᵢ                      // Store complete 8-var chunk
    end for
    
    return C

═══════════════════════════════════════════════════════════════════
PHASE 2: 3D MINIMIZATION PER CHUNK
═══════════════════════════════════════════════════════════════════

Function: SOLVE-CHUNKS(C, form)
Input: C - set of 8-variable chunks
       form - SOP or POS
Output: R - dictionary mapping chunk identifiers to pattern sets

    R ← ∅
    
    for each χ ∈ C do
        // Create 8-variable minimizer for this chunk
        S₈ ← KMAPSOLVER-3D(8, K^χ)
        
        // Apply 3D minimization algorithm (see 3D algorithm document)
        Π, _ ← 3D-MINIMIZE(S₈, form)
        
        // Extract bit patterns (8 bits each with possible don't cares)
        P ← ∅
        for each τ ∈ Π do
            π ← TERM-TO-PATTERN(τ, 8)    // Convert term to bit pattern
            P ← P ∪ {π}
        end for
        
        R[χ] ← P                          // Store patterns for this chunk
    end for
    
    return R

═══════════════════════════════════════════════════════════════════
PHASE 3: 4D CLUSTER IDENTIFICATION
═══════════════════════════════════════════════════════════════════

Function: IDENTIFY-4D-CLUSTERS(R)
Input: R - dictionary mapping chunks to their patterns
Output: Γ₄ᴰ - valid 4D clusters (patterns spanning multiple chunks)
        Γ₃ᴰ - eliminated 3D-only patterns

    // Build inverse mapping: which chunks contain each pattern
    Θ ← ∅
    
    for each (χ, P) ∈ R do
        for each π ∈ P do
            Θ[π] ← Θ[π] ∪ {χ}            // Add chunk to pattern's list
        end for
    end for
    
    // Filter patterns based on chunk distribution
    Γ₄ᴰ ← ∅                              // Valid 4D clusters
    Γ₃ᴰ ← ∅                              // 3D-only patterns (no span merging)
    
    for each (π, X) ∈ Θ do
        if |X| = 1 then
            // Pattern appears in only one chunk: pure 3D cluster
            Γ₃ᴰ ← Γ₃ᴰ ∪ {(π, X)}
            
        else if ¬ADJACENT-CHUNKS(X) then
            // Pattern spans non-adjacent chunks: cannot merge
            Γ₃ᴰ ← Γ₃ᴰ ∪ {(π, χ) : χ ∈ X}
            
        else
            // Pattern spans adjacent chunks: valid 4D cluster
            Γ₄ᴰ[π] ← X
        end if
    end for
    
    // Safety fallback: if no 4D clusters found, treat all as 4D
    if Γ₄ᴰ = ∅ then
        Γ₄ᴰ ← Θ
    end if
    
    return Γ₄ᴰ, Γ₃ᴰ

───────────────────────────────────────────────────────────────────

Function: ADJACENT-CHUNKS(X)
Input: X - set of chunk identifiers
Output: Boolean - true if any pair has Hamming distance = 1

    // Check for Gray code adjacency in span dimension
    for each χᵢ, χⱼ ∈ X, i < j do
        if HAMMING-DISTANCE(χᵢ, χⱼ) = 1 then
            return TRUE                  // Adjacent chunks found
        end if
    end for
    return FALSE                         // No adjacent pairs

═══════════════════════════════════════════════════════════════════
PHASE 4: SPAN-WISE MERGING (Quine-McCluskey)
═══════════════════════════════════════════════════════════════════

Function: SPAN-WISE-MERGE(Γ₄ᴰ)
Input: Γ₄ᴰ - valid 4D clusters
Output: Λ - merged 4D clusters with full metadata

    Λ ← ∅
    
    for each (π, X) ∈ Γ₄ᴰ do
        // Apply Quine-McCluskey to merge chunk identifiers
        Φ ← QUINE-MCCLUSKEY-COMPLETE(X)
        
        for each φ ∈ Φ do
            // Create 4D cluster object
            λ ← {
                inner_pattern: π,                 // 8-bit pattern (depth+breadth+length)
                chunk_pattern: φ,                 // c-bit pattern (span) with don't cares
                full_pattern: φ ∘ π,              // Complete n-bit pattern
                span: 2^(COUNT-DASHES(φ)),        // Number of chunks covered
                cells_3D: GET-3D-COVERAGE(π)      // 3D spatial coverage
            }
            Λ ← Λ ∪ {λ}
        end for
    end for
    
    return Λ

───────────────────────────────────────────────────────────────────

Function: QUINE-MCCLUSKEY-COMPLETE(M)
Input: M - set of minterms (binary strings)
Output: Ψ - minimal set of prime implicants

    // Handle trivial cases
    if |M| = 0 then return ∅
    if |M| = 1 then return M
    
    // Phase 1: Find all prime implicants via iterative merging
    T ← M                                // Current terms
    Π ← ∅                                // Prime implicants
    
    while TRUE do
        T' ← ∅                           // Next iteration terms
        U ← ∅                            // Terms used in merges
        
        for each τᵢ, τⱼ ∈ T, i < j do
            d ← τᵢ ⊕ τⱼ                   // XOR to find differing bits
            
            if POPCOUNT(d) = 1 then
                // Exactly one bit differs: valid merge
                τ' ← τᵢ ∧ τⱼ with bit d set to '-'
                T' ← T' ∪ {τ'}
                U ← U ∪ {τᵢ, τⱼ}          // Mark terms as used
            end if
        end for
        
        // Unused terms cannot be merged further: they are prime implicants
        Π ← Π ∪ (T \ U)
        
        if T' = ∅ then break             // No more merges possible
        T ← T'
    end while
    
    // Phase 2: Select essential prime implicants
    Ψ ← SELECT-ESSENTIAL-PIS(Π, M)
    
    return Ψ

───────────────────────────────────────────────────────────────────

Function: SELECT-ESSENTIAL-PIS(Π, M)
Input: Π - set of prime implicants
       M - set of minterms to cover
Output: Ψ - minimal set of essential prime implicants

    // Build coverage table: which minterms each PI covers
    Cov : Π → 2^M
    for each π ∈ Π do
        Cov[π] ← {m ∈ M : π covers m}
    end for
    
    Ψ ← ∅                                // Essential PIs
    V ← ∅                                // Covered minterms
    
    // Step 1: Find essential prime implicants
    // (minterms covered by only one PI)
    for each m ∈ M do
        Πₘ ← {π ∈ Π : m ∈ Cov[π]}
        
        if |Πₘ| = 1 then
            π ← the unique element in Πₘ  // This PI is essential
            Ψ ← Ψ ∪ {π}
            V ← V ∪ Cov[π]
        end if
    end for
    
    // Step 2: Greedy cover for remaining uncovered minterms
    R ← M \ V                            // Remaining uncovered
    Π' ← Π \ Ψ                           // Remaining PIs
    
    while R ≠ ∅ and Π' ≠ ∅ do
        // Select PI covering most uncovered minterms
        π* ← arg max_{π ∈ Π'} |Cov[π] ∩ R|
        
        if |Cov[π*] ∩ R| = 0 then break  // No more coverage possible
        
        Ψ ← Ψ ∪ {π*}
        V ← V ∪ Cov[π*]
        R ← R \ Cov[π*]
        Π' ← Π' \ {π*}
    end while
    
    return Ψ

═══════════════════════════════════════════════════════════════════
PHASE 5: ESSENTIAL PRIME IMPLICANT SELECTION (Span Dominance)
═══════════════════════════════════════════════════════════════════

Function: SELECT-EPIS-BY-SPAN(Λ)
Input: Λ - merged 4D clusters
Output: Ψ - essential prime implicants selected by span dominance

    // Group clusters by their inner pattern (8-bit pattern)
    G ← ∅
    for each λ ∈ Λ do
        G[λ.inner_pattern] ← G[λ.inner_pattern] ∪ {λ}
    end for
    
    Ψ ← ∅
    
    for each (π, L) ∈ G do
        // Further group by 3D cell coverage
        H ← ∅
        for each λ ∈ L do
            H[λ.cells_3D] ← H[λ.cells_3D] ∪ {λ}
        end for
        
        // For each cell group, select clusters with maximum span
        for each (ω, L') ∈ H do
            s_max ← max{λ.span : λ ∈ L'}  // Find maximum span
            
            for each λ ∈ L' do
                if λ.span = s_max then
                    Ψ ← Ψ ∪ {λ}          // Maximum span → essential
                end if
            end for
        end for
    end for
    
    return Ψ

───────────────────────────────────────────────────────────────────

Dominance Criterion:
    λ₁ dominates λ₂ ⟺ 
        (λ₁.inner_pattern = λ₂.inner_pattern) ∧
        (λ₁.cells_3D = λ₂.cells_3D) ∧
        (λ₁.span > λ₂.span)
    
    Interpretation: A cluster with greater span covering the same 3D
    cells with the same inner pattern dominates smaller-span clusters

═══════════════════════════════════════════════════════════════════
PHASE 6: COVERAGE VERIFICATION & COMPLETION
═══════════════════════════════════════════════════════════════════

// The greedy coverage used here was found to be inconsequential on the final result
// and was eliminated from the python implementation. Users seeking to replicate the algorithm  
// are at liberty to determine if to include this fallback or not. 

Function: VERIFY-AND-COMPLETE(Ψ, K, Λ)
Input: Ψ - selected EPIs
       K - original K-map structure
       Λ - all available clusters
Output: Ψ - complete coverage EPIs (with additions if needed)

    // Get all minterms where function = 1
    M_all ← {(ι, ρ) : K[ι][ρ] = 1, ∀ι, ρ}
    
    // Get minterms covered by selected EPIs
    M_cov ← ⋃_{λ ∈ Ψ} EXPAND-PATTERN(λ.full_pattern)
    
    // Find uncovered minterms
    M_unc ← M_all \ M_cov
    
    if M_unc ≠ ∅ then
        // Attempt greedy coverage with remaining clusters
        Ψ_add ← GREEDY-COVER-4D(M_unc, Λ, Ψ)
        Ψ ← Ψ ∪ Ψ_add
        
        // Re-verify coverage
        M_cov ← ⋃_{λ ∈ Ψ} EXPAND-PATTERN(λ.full_pattern)
        M_unc ← M_all \ M_cov
        
        if M_unc ≠ ∅ then
            ERROR: "Incomplete coverage - isolated regions exist"
        end if
    end if
    
    return Ψ

───────────────────────────────────────────────────────────────────

Function: GREEDY-COVER-4D(M_unc, Λ, Ψ)
Input: M_unc - uncovered minterms
       Λ - all available clusters
       Ψ - already selected EPIs
Output: Ψ_add - additional clusters to cover M_unc

    Ψ_add ← ∅
    R ← M_unc                            // Remaining uncovered
    Λ' ← Λ \ {λ : λ ∈ Ψ}                 // Available clusters (not yet used)
    
    while R ≠ ∅ and Λ' ≠ ∅ do
        // Find cluster covering most remaining minterms
        λ* ← arg max_{λ ∈ Λ'} |EXPAND-PATTERN(λ.full_pattern) ∩ R|
        
        if |EXPAND-PATTERN(λ*.full_pattern) ∩ R| = 0 then
            break                        // No more coverage possible
        end if
        
        Ψ_add ← Ψ_add ∪ {λ*}
        R ← R \ EXPAND-PATTERN(λ*.full_pattern)
        Λ' ← Λ' \ {λ*}
    end while
    
    return Ψ_add

───────────────────────────────────────────────────────────────────

Function: EXPAND-PATTERN(π)
Input: π - bit pattern with don't cares
Output: M - set of all concrete minterms matching π

    // Expand don't cares to all possible combinations
    d ← COUNT-DASHES(π)
    M ← ∅
    
    for i ← 0 to 2^d - 1 do
        σ ← SUBSTITUTE-DASHES(π, BINARY(i, d))
        M ← M ∪ {σ}
    end for
    
    return M

═══════════════════════════════════════════════════════════════════
PHASE 7: EXPRESSION GENERATION
═══════════════════════════════════════════════════════════════════

Function: GENERATE-EXPRESSION(Ψ, form)
Input: Ψ - essential prime implicants
       form - SOP or POS
Output: T - set of terms
        E - expression string

    T ← ∅
    
    // Convert each cluster to a Boolean term
    for each λ ∈ Ψ do
        τ ← BITS-TO-TERM(λ.full_pattern, form)
        T ← T ∪ {τ}
    end for
    
    // Join terms with appropriate operator
    if form = SOP then
        E ← JOIN(T, " + ")               // Sum of products
    else if form = POS then
        E ← JOIN(T, " · ")               // Product of sums
    end if
    
    return T, E

═══════════════════════════════════════════════════════════════════
MAIN ALGORITHM
═══════════════════════════════════════════════════════════════════

Function: 4D-MINIMIZE(K, n, form)
Input: K - n-variable K-map structure
       n - number of variables (n > 8)
       form - SOP or POS
Output: T - set of terms
        E - minimized Boolean expression

    // Fallback to 3D algorithm for n ≤ 8
    if n ≤ 8 then
        return 3D-MINIMIZE(K, form)
    end if
    
    // Phase 1: Partition into 8-variable chunks
    C ← PARTITION-INTO-CHUNKS(K, n)
    
    // Phase 2: Apply 3D minimization to each chunk
    R ← SOLVE-CHUNKS(C, form)
    
    // Phase 3: Identify valid 4D clusters (patterns spanning chunks)
    Γ₄ᴰ, Γ₃ᴰ ← IDENTIFY-4D-CLUSTERS(R)
    
    // Phase 4: Merge chunk identifiers in span dimension
    Λ ← SPAN-WISE-MERGE(Γ₄ᴰ)
    
    // Phase 5: Select EPIs using span dominance criterion
    Ψ ← SELECT-EPIS-BY-SPAN(Λ)
    
    // Phase 6: Verify complete coverage and add missing terms
    Ψ ← VERIFY-AND-COMPLETE(Ψ, K, Λ)
    
    // Phase 7: Generate final Boolean expression
    T, E ← GENERATE-EXPRESSION(Ψ, form)
    
    return T, E

═══════════════════════════════════════════════════════════════════
COMPLEXITY ANALYSIS
═══════════════════════════════════════════════════════════════════

Time Complexity:
    Chunk partitioning:           O(2^n)
    3D minimization per chunk:    O(2^c × f(8)) where f(8) is 3D cost
    4D cluster identification:    O(|patterns| × |chunks|)
    Span-wise merging:            O(|Γ₄ᴰ| × 2^(2c))
    EPI selection:                O(|Λ|²)
    Coverage verification:        O(2^n)
    
Overall: O(2^n + |Γ₄ᴰ| × 2^(2c))
    where c = n - 8, typically |Γ₄ᴰ| << 2^n

Space Complexity: O(2^n) for storing K-map chunks

Optimality:
    3D minimization per chunk:    Exact (via 3D algorithm)
    Span-wise merging:            Exact (Quine-McCluskey)
    Span dominance:               Maximal grouping across chunks
    Coverage completion:          Greedy (if needed)

═══════════════════════════════════════════════════════════════════
END OF ALGORITHM
═══════════════════════════════════════════════════════════════════
```