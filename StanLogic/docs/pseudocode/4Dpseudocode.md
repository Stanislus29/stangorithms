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
PHASE 6: COVERAGE VERIFICATION & 3D PATTERN COLLECTION
═══════════════════════════════════════════════════════════════════

Algorithm: VERIFY-AND-COLLECT-3D(Ψ, R, K, M_all)
Input: Ψ - selected 4D EPIs (clusters spanning chunks)
       R - chunk results dictionary (identifier → patterns)
       K - original K-map structure
       M_all - all target minterms
Output: Π_all - combined set of 4D and 3D patterns

    // Separate 4D patterns from 3D chunk patterns
    Π₄ᴰ ← {λ.full_pattern : λ ∈ Ψ}
    
    // Calculate 4D pattern coverage
    M_cov₄ᴰ ← ⋃_{π ∈ Π₄ᴰ} EXPAND-PATTERN(π)
    M_unc ← M_all \ M_cov₄ᴰ
    
    print("4D clusters: ", |Π₄ᴰ|, " patterns")
    print("4D coverage: ", |M_cov₄ᴰ|, "/", |M_all|, " minterms")
    print("Uncovered after 4D: ", |M_unc|, " minterms")
    
    // Collect 3D patterns for uncovered minterms
    Π₃ᴰ ← COLLECT-3D-FOR-COVERAGE(R, Π₄ᴰ, M_unc, M_all)
    
    // Calculate 3D coverage
    M_cov₃ᴰ ← ⋃_{π ∈ Π₃ᴰ} EXPAND-PATTERN(π)
    M_unc ← M_unc \ M_cov₃ᴰ
    
    print("3D patterns: ", |Π₃ᴰ|, " additional patterns")
    print("3D coverage: ", |M_cov₃ᴰ|, " additional minterms")
    print("Uncovered after 3D: ", |M_unc|, " minterms")
    
    // Fallback coverage for remaining uncovered
    if M_unc ≠ ∅ then
        Π_fallback ← FALLBACK-COVERAGE(M_unc)
        print("Fallback: ", |Π_fallback|, " direct minterm patterns")
    else
        Π_fallback ← ∅
    end if
    
    // Combine all patterns
    Π_all ← Π₄ᴰ ∪ Π₃ᴰ ∪ Π_fallback
    
    return Π_all

───────────────────────────────────────────────────────────────────

Function: COLLECT-3D-FOR-COVERAGE(R, Π₄ᴰ, M_unc, M_all)
Input: R - chunk results dictionary (identifier → patterns)
       Π₄ᴰ - set of 4D patterns already selected
       M_unc - uncovered minterms
       M_all - all target minterms
Output: Π₃ᴰ - set of 3D patterns for coverage

    if M_unc = ∅ then
        return ∅                         // All covered by 4D patterns
    end if
    
    // Build list of 3D candidate patterns from chunks
    candidates ← []
    
    for each (δ, patterns) ∈ R do
        for each π ∈ patterns do
            π_full ← δ ∘ π               // Full pattern (chunk ID + inner)
            
            if π_full ∈ Π₄ᴰ then
                continue                 // Skip 4D patterns
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
    
    // Find essential 3D patterns
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
PHASE 7: FINAL QUINE-MCCLUSKEY OPTIMIZATION
═══════════════════════════════════════════════════════════════════

Algorithm: OPTIMIZE-WITH-QUINE-MCCLUSKEY(Π_all, M_all)
Input: Π_all - set of all patterns (4D + 3D + fallback)
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
    
This allows patterns from different sources (4D, 3D, fallback) to be
merged and optimized together, producing a globally minimal result.

═══════════════════════════════════════════════════════════════════
PHASE 8: EXPRESSION GENERATION
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
    
    // Get all target minterms
    M_all ← GET-ALL-MINTERMS(K)
    
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
    
    // Phase 6: Coverage verification and 3D pattern collection
    Π_all ← VERIFY-AND-COLLECT-3D(Ψ, R, K, M_all)
    
    // Phase 7: Final global optimization
    Π_optimal ← OPTIMIZE-WITH-QUINE-MCCLUSKEY(Π_all, M_all)
    
    // Phase 8: Generate final Boolean expression
    T, E ← GENERATE-EXPRESSION(Π_optimal, form)
    
    return T, E

End Algorithm

═══════════════════════════════════════════════════════════════════
END OF ALGORITHM
═══════════════════════════════════════════════════════════════════
```