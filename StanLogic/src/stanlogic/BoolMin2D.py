"""
A Library for minimizing 2D K-maps (up to 4 variables) using
the Karnaugh Mapping method with bitmask optimizations.

Author: Somtochukwu Stanislus Emek-Onwuneme
Date: 8th September, 2025
""" 

# Author: Somtochukwu Stanislus Emeka-Onwuneme 
# Publication Date: 8th September, 2025
# Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme
#---------------------------------------------------------

from collections import defaultdict

class BoolMin2D:
    def __init__(self, kmap, convention = "vranseic"):
        """
        Initialize K-map solver with input matrix and variable ordering convention.
        
        Args:
            kmap: 2D list representing K-map values (0, 1, or 'd' for don't care)
            convention: Variable ordering - "vranseic" (default, cols=x1x2) or "mano_kime" (rows=x1x2)
        """
        # Store input K-map and dimensions
        self.kmap = kmap
        self.num_rows = len(kmap)
        self.num_cols = len(kmap[0])
        self.convention = convention.lower().strip()

        # Determine K-map size and set up variable labeling
        size = self.num_rows * self.num_cols
        if size == 4:  # 2-variable K-map (2x2)
            self.num_vars = 2
            self.row_labels = ["0", "1"]
            self.col_labels = ["0", "1"]
        elif size == 8:  # 3-variable K-map (2x4)
            self.num_vars = 3
            self.row_labels = ["0", "1"]
            # Gray code ordering for 3-var K-map columns
            self.col_labels = ["00", "01", "11", "10"]
        elif size == 16:  # 4-variable K-map (4x4)
            self.num_vars = 4
            # Gray code ordering for both rows and columns
            self.row_labels = ["00", "01", "11", "10"]
            self.col_labels = ["00", "01", "11", "10"]
        else:
            raise ValueError("K-map must be 2x2 (2 vars), 2x4/4x2 (3 vars), or 4x4 (4 vars)")

        # Define possible group sizes (all powers of 2)
        # Format: (height, width) of rectangular groups
        self.group_sizes = [(1,1),(1,2),(2,1),(2,2),
                           (1,4),(4,1),(2,4),(4,2),(4,4)]

        # Filter out group sizes that don't fit this K-map's dimensions
        self.group_sizes = [(h,w) for (h,w) in self.group_sizes 
                           if h <= self.num_rows and w <= self.num_cols]

        # Precompute lookup tables for efficiency
        # _cell_index: maps (row,col) to minterm/maxterm index
        # _cell_bits: maps (row,col) to binary representation
        self._cell_index = [[self._cell_to_term(r, c) for c in range(self.num_cols)] 
                           for r in range(self.num_rows)]
        self._cell_bits = [[self._cell_to_bits(r, c) for c in range(self.num_cols)] 
                          for r in range(self.num_rows)]
        
        # Reverse lookup: map term index back to (row,col)
        self._index_to_rc = {self._cell_index[r][c]: (r, c) 
                            for r in range(self.num_rows) 
                            for c in range(self.num_cols)}

    def _mask_to_coords(self, mask):
        """Helper to convert a bitmask to a list of [row, col] coordinates."""
        coords = []
        temp = mask
        while temp:
            low = temp & -temp
            idx = low.bit_length() - 1
            temp -= low
            if idx in self._index_to_rc:
                coords.append(self._index_to_rc[idx])
        return coords

    def _cell_to_term(self, r, c):
        """
        Convert K-map cell coordinates to minterm/maxterm index.
        Uses concatenated binary strings from row and column labels.
        """
        bits = self.col_labels[c] + self.row_labels[r]
        return int(bits, 2)

    def _cell_to_bits(self, r, c):
        """
        Get binary representation of cell based on chosen variable convention.
        
        Vranesic: cols represent x1x2, rows represent x3x4
        Mano-Kime: rows represent x1x2, cols represent x3x4
        """
        if self.convention == "mano_kime":
            bits = self.row_labels[r] + self.col_labels[c]
        else:  # vranesic (default)
            bits = self.col_labels[c] + self.row_labels[r]
        return bits

    def _get_group_coords(self, r, c, h, w):
        """
        Generate list of cell coordinates for a group starting at (r,c).
        Handles K-map wrapping (adjacent edges are considered adjacent).
        """
        return [((r+i) % self.num_rows, (c+j) % self.num_cols)
                for i in range(h) for j in range(w)]

    def find_all_groups(self, allow_dontcare=False):
        """
        Find all valid groups of 1's (and optionally don't cares).
        Returns list of bitmasks, where each bit represents a cell's inclusion.
        """
        groups = set()
        # Try all possible group positions and sizes
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                for h, w in self.group_sizes:
                    coords = self._get_group_coords(r, c, h, w)
                    
                    # Check if this forms a valid group
                    if allow_dontcare:
                        # Group must contain at least one 1 and only 1's or don't cares
                        if all(self.kmap[rr][cc] in (1, 'd') for rr, cc in coords) and \
                           any(self.kmap[rr][cc] == 1 for rr, cc in coords):
                            # Create bitmask representation of group
                            mask = 0
                            for rr, cc in coords:
                                mask |= 1 << self._cell_index[rr][cc]
                            groups.add(mask)
                    else:
                        # Group must contain only 1's
                        if all(self.kmap[rr][cc] == 1 for rr, cc in coords):
                            mask = 0
                            for rr, cc in coords:
                                mask |= 1 << self._cell_index[rr][cc]
                            groups.add(mask)
        return list(groups)

    def find_all_groups_pos(self, allow_dontcare=False):
        """
        Find all valid groups of 0's (and optionally don't cares) for Product of Sums form.
        Similar to find_all_groups but looks for 0's instead of 1's.
        Returns list of bitmasks representing each valid group.
        """
        groups = set()
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                for h, w in self.group_sizes:
                    coords = self._get_group_coords(r, c, h, w)

                    # For POS, we group 0's (and optionally don't cares)
                    if allow_dontcare:
                        # Group must have at least one 0 and only 0's or don't cares
                        if all(self.kmap[rr][cc] in (0, 'd') for rr, cc in coords) and \
                           any(self.kmap[rr][cc] == 0 for rr, cc in coords):
                            # Create bitmask for the group
                            mask = 0
                            for rr, cc in coords:
                                mask |= 1 << self._cell_index[rr][cc]
                            groups.add(mask)
                    else:
                        # Group must contain only 0's
                        if all(self.kmap[rr][cc] == 0 for rr, cc in coords):
                            mask = 0
                            for rr, cc in coords:
                                mask |= 1 << self._cell_index[rr][cc]
                            groups.add(mask)
        return list(groups)
    
    def filter_prime_implicants(self, groups):
        """
        Remove redundant groups that are completely covered by larger groups.
        Uses bitmask operations for efficient subset checking.
        
        Args:
            groups: List of integer bitmasks representing K-map groups
            
        Returns:
            List of prime implicant bitmasks (non-redundant groups)
        """
        primes = []
        # Sort by size (bit count) descending for early pruning
        groups_sorted = sorted(groups, key=lambda g: g.bit_count(), reverse=True)
        
        for i, g in enumerate(groups_sorted):
            is_subset = False
            # Check if g is a subset of any other group
            for other in groups_sorted:
                if other == g:
                    continue
                # Bitwise AND: if g is subset of other, (g & other) == g
                if (g & other) == g:
                    is_subset = True
                    break
            if not is_subset:
                primes.append(g)
        return primes

    # ---------- Boolean simplification ---------- #
    def _simplify_group_bits(self, bits_list):
        """
        Simplify a group of bits (from a K-map group) to a Boolean term.
        Compares each bit position across the group, using '-' for varying bits.
        
        Args:
            bits_list: List of bit strings representing the group
            
        Returns:
            Simplified Boolean term as a string
        """
        bits = list(bits_list[0])
        for b in bits_list[1:]:
            for i in range(self.num_vars):
                if bits[i] != b[i]:
                    bits[i] = '-'

        vars_ = [f"x{i+1}" for i in range(self.num_vars)]
        term = []
        for i, b in enumerate(bits):
            if b == '0':
                term.append(vars_[i] + "'")
            elif b == '1':
                term.append(vars_[i])
        return "".join(term)
    
    def _simplify_group_bits_pos(self, bits_list):
        """Convert group bits to POS term."""
        bits = list(bits_list[0])
        for b in bits_list[1:]:
            for i in range(self.num_vars):
                if bits[i] != b[i]:
                    bits[i] = '-'

        vars_ = [f"x{i+1}" for i in range(self.num_vars)]
        term = []
        for i, b in enumerate(bits):
            if b == '1':
                term.append(vars_[i] + "'")
            elif b == '0':
                term.append(vars_[i])
        return "(" + " + ".join(term) + ")"

    # ---------- Main minimization ---------- #
    def minimize(self, form='sop'):
        """
        Minimize K-map expression using Quine-McCluskey algorithm with bitmask optimizations.
        
        Algorithm steps:
        1. Find all valid groups (rectangular power-of-2 sized)
        2. Filter to prime implicants (non-redundant groups)
        3. Compute term expressions and coverage patterns
        4. Find essential prime implicants
        5. Use greedy set cover for remaining terms
        6. Remove any remaining redundancy
        7. Verify complete coverage of all target minterms
        
        Args:
            form: 'sop' for Sum of Products or 'pos' for Product of Sums
            
        Returns:
            tuple: (list of minimized terms, complete expression string)
        """
        if form.lower() not in ['sop', 'pos']:
            raise ValueError("form must be either 'sop' or 'pos'")

        # For POS, we group 0's; for SOP, we group 1's
        target_val = 0 if form.lower() == 'pos' else 1
        
        # OPTIMIZATION 1: Check for constant functions (all same value)
        # Collect all defined (non-don't-care) cells
        defined_cells = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.kmap[r][c] != 'd':
                    defined_cells.append(self.kmap[r][c])
        
        # Handle edge cases for constant functions
        if not defined_cells:  # All don't cares
            # Return 0 for SOP, 1 for POS (arbitrary choice for don't care)
            if form.lower() == 'sop':
                return [], "0"
            else:
                return [], "1"
        
        # Check if all defined cells are 0 (no target values)
        if all(cell == (1 - target_val) for cell in defined_cells):
            # All zeros for SOP or all ones for POS: return constant
            if form.lower() == 'sop':
                return [], "0"
            else:
                return [], "1"
        
        # Check if all defined cells equal target value
        if all(cell == target_val for cell in defined_cells):
            # All ones for SOP or all zeros for POS: return constant 1 or 0
            if form.lower() == 'sop':
                return ["1"], "1"
            else:
                return ["0"], "0"

        # Select appropriate grouping and term formatting methods
        if form.lower() == 'pos':
            groups = self.find_all_groups_pos(allow_dontcare=True)
            simplify_method = self._simplify_group_bits_pos
            join_operator = " * "  # POS terms are ANDed
        else:
            groups = self.find_all_groups(allow_dontcare=True)
            simplify_method = self._simplify_group_bits
            join_operator = " + "  # SOP terms are ORed

        # Find prime implicants (non-redundant groups)
        prime_groups = self.filter_prime_implicants(groups)

        # Generate terms and track their coverage
        # CRITICAL: Coverage mask ONLY includes cells with TARGET VALUE (not don't-cares)
        # This ensures we track actual minterms/maxterms that must be covered
        prime_terms = []
        prime_covers = []
        prime_terms_bits = []  # Store bit patterns for later validation
        
        for gmask in prime_groups:
            cover_mask = 0  # Tracks ONLY cells with target value
            bits_list = []  # Collects binary representations FROM TARGET CELLS ONLY
            has_target = False  # Track if group covers any target cells
            
            # Process each cell in the group
            temp = gmask
            while temp:
                # Extract least significant 1-bit
                low = temp & -temp
                idx = low.bit_length() - 1
                temp -= low
                
                # Map bit position back to K-map coordinates
                if idx not in self._index_to_rc:
                    continue  # Skip invalid indices
                    
                r, c = self._index_to_rc[idx]
                
                # CRITICAL FIX: Include in coverage AND bit pattern ONLY if cell has target value
                # Don't-care cells can be part of the group (for size) but should NOT
                # affect the term simplification!
                if self.kmap[r][c] == target_val:
                    cover_mask |= 1 << idx
                    bits_list.append(self._cell_bits[r][c])  # Only use target cell bits!
                    has_target = True
                
            # OPTIMIZATION 2: Validate that group covers at least one target cell
            # Groups with only don't-cares should be skipped
            if not has_target or cover_mask == 0:
                continue
            
            # OPTIMIZATION 3: Ensure bits_list is non-empty before simplification
            if not bits_list or len(bits_list) == 0:
                continue
                
            # Generate Boolean term from bit pattern
            term_str = simplify_method(bits_list)
            
            # OPTIMIZATION 4: Skip invalid/empty terms
            if not term_str or term_str.strip() == "":
                continue
                
            # Store term, coverage, and bit pattern
            prime_terms.append(term_str)
            prime_covers.append(cover_mask)
            prime_terms_bits.append(bits_list)

        # Build coverage lookup: which primes cover each minterm
        minterm_to_primes = defaultdict(list)
        all_minterms_mask = 0
        for p_idx, cover in enumerate(prime_covers):
            all_minterms_mask |= cover
            temp = cover
            while temp:
                low = temp & -temp
                idx = low.bit_length() - 1
                temp -= low
                minterm_to_primes[idx].append(p_idx)

        # Find essential prime implicants (only coverage for some minterm)
        essential_indices = set()
        for m, primes in minterm_to_primes.items():
            if len(primes) == 1:
                essential_indices.add(primes[0])

        # Track coverage achieved by essential primes
        covered_mask = 0
        for i in essential_indices:
            covered_mask |= prime_covers[i]

        # OPTIMIZATION 4: Enhanced greedy set cover with exhaustive coverage validation
        remaining_mask = all_minterms_mask & ~covered_mask
        selected = set(essential_indices)
        
        # Track iterations to prevent infinite loops
        max_iterations = len(prime_covers) * 3  # Allow more iterations for complex cases
        iteration = 0
        
        # Greedy selection loop
        while remaining_mask and iteration < max_iterations:
            iteration += 1
            
            # Find prime implicant covering most uncovered minterms
            best_idx, best_cover_count = None, -1
            for idx in range(len(prime_covers)):
                if idx in selected:
                    continue
                # Count NEW bits this prime would cover
                cover = prime_covers[idx] & remaining_mask
                count = cover.bit_count()
                if count > best_cover_count:
                    best_cover_count = count
                    best_idx = idx
            
            # Termination check: no prime covers any remaining minterm
            if best_idx is None or best_cover_count == 0:
                # CRITICAL: Try to force coverage of remaining minterms
                # This handles cases where greedy algorithm gets stuck
                forced_coverage = False
                for idx in range(len(prime_covers)):
                    if idx not in selected:
                        overlap = prime_covers[idx] & remaining_mask
                        if overlap:  # This prime covers at least one uncovered minterm
                            selected.add(idx)
                            covered_mask |= prime_covers[idx]
                            remaining_mask = all_minterms_mask & ~covered_mask
                            forced_coverage = True
                            break
                
                # If we forced coverage, continue the loop
                if forced_coverage:
                    continue
                else:
                    # No prime can cover remaining minterms - break
                    break
            
            # Add best prime and update coverage
            selected.add(best_idx)
            covered_mask |= prime_covers[best_idx]
            remaining_mask = all_minterms_mask & ~covered_mask

        # Remove any redundant selected terms
        def covers_with_indices(indices):
            """Helper: get combined coverage mask for given term indices."""
            mask = 0
            for i in indices:
                mask |= prime_covers[i]
            return mask

        chosen = set(selected)
        for idx in list(sorted(chosen)):
            # Try removing each term; keep removal if coverage maintained
            trial = chosen - {idx}
            if covers_with_indices(trial) == covers_with_indices(chosen):
                chosen = trial

        # Build final minimized expression
        final_terms = [prime_terms[i] for i in sorted(chosen)]
        
        # OPTIMIZATION 5: Exhaustive coverage verification with forced complete coverage
        # Build expected coverage mask from ALL cells with target value
        expected_coverage = 0
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if self.kmap[r][c] == target_val:
                    idx = self._cell_index[r][c]
                    expected_coverage |= (1 << idx)
        
        # CRITICAL: Verify complete coverage
        if expected_coverage != 0 and covered_mask != expected_coverage:
            # Coverage is incomplete - perform exhaustive recovery
            still_uncovered = expected_coverage & ~covered_mask
            fallback_selected = set(chosen)
            
            # Strategy 1: Add primes that cover uncovered minterms (greedy recovery)
            attempts = 0
            max_attempts = len(prime_covers)
            
            while still_uncovered and attempts < max_attempts:
                attempts += 1
                best_recovery_idx = None
                best_recovery_count = 0
                
                # Find prime that covers most uncovered minterms
                for idx in range(len(prime_covers)):
                    if idx not in fallback_selected:
                        overlap = prime_covers[idx] & still_uncovered
                        count = overlap.bit_count()
                        if count > best_recovery_count:
                            best_recovery_count = count
                            best_recovery_idx = idx
                
                if best_recovery_idx is not None and best_recovery_count > 0:
                    fallback_selected.add(best_recovery_idx)
                    covered_mask |= prime_covers[best_recovery_idx]
                    still_uncovered = expected_coverage & ~covered_mask
                else:
                    # No single prime helps - try adding ALL remaining primes
                    for idx in range(len(prime_covers)):
                        if idx not in fallback_selected:
                            fallback_selected.add(idx)
                            covered_mask |= prime_covers[idx]
                    still_uncovered = expected_coverage & ~covered_mask
                    break
            
            # Update selection
            chosen = fallback_selected
            final_terms = [prime_terms[i] for i in sorted(chosen)]
        
        # OPTIMIZATION 6: Handle empty results properly
        if not final_terms:
            # If no terms but we had target values, something went wrong
            # Return appropriate constant
            if expected_coverage != 0:
                # This should not happen - indicates algorithm failure
                # Return tautology to ensure coverage
                if form.lower() == 'sop':
                    return ["1"], "1"
                else:
                    return ["0"], "0"
            else:
                # No target values: return constant
                if form.lower() == 'sop':
                    return [], "0"
                else:
                    return [], "1"
        
        # Return final expression
        expression = join_operator.join(final_terms)
        return final_terms, expression

    def minimize_visualize(self, form='sop'):
        """
        Minimizes the K-map and returns a detailed step-by-step breakdown for visualization.
        """
        if form.lower() not in ['sop', 'pos']:
            raise ValueError("form must be either 'sop' or 'pos'")

        steps = {}
        target_val = 0 if form.lower() == 'pos' else 1

        # Step 1: Find all valid groups
        if form.lower() == 'pos':
            groups = self.find_all_groups_pos(allow_dontcare=True)
            simplify_method = self._simplify_group_bits_pos
            join_operator = " * "
        else:
            groups = self.find_all_groups(allow_dontcare=True)
            simplify_method = self._simplify_group_bits
            join_operator = " + "
        
        steps['allGroups'] = {
            'count': len(groups),
            'masks': groups,
            'coords': [self._mask_to_coords(g) for g in groups]
        }

        # Step 2: Filter to prime implicants
        prime_groups = self.filter_prime_implicants(groups)
        prime_terms_map = {}
        for g in prime_groups:
            bits_list = [self._cell_bits[r][c] for r, c in self._mask_to_coords(g)]
            prime_terms_map[g] = simplify_method(bits_list) if bits_list else ""

        steps['primeImplicants'] = {
            'count': len(prime_groups),
            'masks': prime_groups,
            'coords': [self._mask_to_coords(g) for g in prime_groups],
            'terms': [prime_terms_map[g] for g in prime_groups]
        }

        # Step 3: Compute coverage for each prime implicant
        prime_covers = []
        for gmask in prime_groups:
            cover_mask = 0
            temp = gmask
            while temp:
                low = temp & -temp
                idx = low.bit_length() - 1
                temp -= low
                r, c = self._index_to_rc[idx]
                if self.kmap[r][c] == target_val:
                    cover_mask |= (1 << idx)
            prime_covers.append(cover_mask)

        steps['primeWithCoverage'] = {
            'coords': steps['primeImplicants']['coords'],
            'terms': steps['primeImplicants']['terms'],
            'coverageCounts': [c.bit_count() for c in prime_covers]
        }

        # Step 4: Find essential primes
        minterm_to_primes = defaultdict(list)
        all_minterms_mask = 0
        for p_idx, cover in enumerate(prime_covers):
            all_minterms_mask |= cover
            temp = cover
            while temp:
                low = temp & -temp; idx = low.bit_length() - 1; temp -= low
                minterm_to_primes[idx].append(p_idx)
        
        essential_indices = {primes[0] for primes in minterm_to_primes.values() if len(primes) == 1}
        
        steps['essentialPrimes'] = {
            'indices': sorted(list(essential_indices)),
            'coords': [self._mask_to_coords(prime_groups[i]) for i in essential_indices],
            'terms': [prime_terms_map[prime_groups[i]] for i in essential_indices]
        }

        # Step 5: Greedy set cover
        covered_mask = 0
        for i in essential_indices:
            covered_mask |= prime_covers[i]
        
        remaining_mask = all_minterms_mask & ~covered_mask
        selected = set(essential_indices)
        greedy_selections = []

        while remaining_mask:
            best_idx, best_cover_count = -1, -1
            for idx in range(len(prime_covers)):
                if idx in selected: continue
                new_coverage = prime_covers[idx] & remaining_mask
                count = new_coverage.bit_count()
                if count > best_cover_count:
                    best_cover_count = count
                    best_idx = idx
            
            if best_idx == -1 or best_cover_count == 0: break
            
            selected.add(best_idx)
            greedy_selections.append({
                'term': prime_terms_map[prime_groups[best_idx]],
                'newCoverage': best_cover_count
            })
            covered_mask |= prime_covers[best_idx]
            remaining_mask = all_minterms_mask & ~covered_mask
        
        steps['greedySelections'] = greedy_selections

        # Step 6: Final redundancy check and expression assembly
        def covers_with_indices(indices):
            mask = 0
            for i in indices: mask |= prime_covers[i]
            return mask

        chosen = set(selected)
        for idx in sorted(list(chosen)):
            trial = chosen - {idx}
            if covers_with_indices(trial) == covers_with_indices(chosen):
                chosen = trial
        
        final_terms = [prime_terms_map[prime_groups[i]] for i in sorted(list(chosen))]
        expression = join_operator.join(final_terms) if final_terms else ('1' if form == 'sop' else '0')

        steps['finalSelected'] = {
            'indices': sorted(list(chosen)),
            'coords': [self._mask_to_coords(prime_groups[i]) for i in chosen],
            'terms': final_terms
        }

        return expression, steps
    
    """
    Additional functions for BoolMin2D to generate output files with:
    - Minimized logic expression
    - Logic gate visualization
    - Verilog code generation
    """

    def generate_verilog(self, module_name="logic_circuit", form='sop'):
        """
        Generate Verilog HDL code for the minimized Boolean expression.
        
        Args:
            module_name: Name of the Verilog module (default: "logic_circuit")
            form: 'sop' for Sum of Products or 'pos' for Product of Sums
            
        Returns:
            String containing complete Verilog module code
        """
        terms, expression = self.minimize(form=form)
        
        # Generate input port list
        inputs = ", ".join([f"x{i+1}" for i in range(self.num_vars)])
        
        # Build Verilog code
        verilog_code = f"""module {module_name}({inputs}, F);
        // Inputs
        input {inputs};
        
        // Output
        output F;
        
        // Minimized expression: {expression}
        """
        
        if form.lower() == 'sop':
            # Generate SOP logic
            if not terms:
                verilog_code += "    assign F = 1'b0;\n"
            elif len(terms) == 1:
                verilog_code += f"    assign F = {self._term_to_verilog(terms[0])};\n"
            else:
                # Multiple terms - create intermediate wires
                verilog_code += f"\n    // Intermediate product terms\n"
                for i, term in enumerate(terms):
                    verilog_code += f"    wire p{i};\n"
                verilog_code += "\n"
                
                for i, term in enumerate(terms):
                    verilog_code += f"    assign p{i} = {self._term_to_verilog(term)};\n"
                
                verilog_code += f"\n    // Sum of products\n"
                sum_terms = " | ".join([f"p{i}" for i in range(len(terms))])
                verilog_code += f"    assign F = {sum_terms};\n"
        else:  # POS
            if not terms:
                verilog_code += "    assign F = 1'b1;\n"
            elif len(terms) == 1:
                verilog_code += f"    assign F = {self._term_to_verilog_pos(terms[0])};\n"
            else:
                # Multiple terms - create intermediate wires
                verilog_code += f"\n    // Intermediate sum terms\n"
                for i, term in enumerate(terms):
                    verilog_code += f"    wire s{i};\n"
                verilog_code += "\n"
                
                for i, term in enumerate(terms):
                    verilog_code += f"    assign s{i} = {self._term_to_verilog_pos(term)};\n"
                
                verilog_code += f"\n    // Product of sums\n"
                prod_terms = " & ".join([f"s{i}" for i in range(len(terms))])
                verilog_code += f"    assign F = {prod_terms};\n"
        
        verilog_code += "\nendmodule"
        return verilog_code

    def _term_to_verilog(self, term):
        """Convert SOP term to Verilog syntax (e.g., "x1x2'x3" -> "x1 & ~x2 & x3")"""
        if not term:
            return "1'b1"
        
        verilog_parts = []
        i = 0
        while i < len(term):
            if term[i] == 'x':
                # Extract variable number
                var_num = ""
                i += 1
                while i < len(term) and term[i].isdigit():
                    var_num += term[i]
                    i += 1
                
                # Check for complement
                if i < len(term) and term[i] == "'":
                    verilog_parts.append(f"~x{var_num}")
                    i += 1
                else:
                    verilog_parts.append(f"x{var_num}")
            else:
                i += 1
        
        return " & ".join(verilog_parts) if verilog_parts else "1'b1"

    def _term_to_verilog_pos(self, term):
        """Convert POS term to Verilog syntax (e.g., "(x1 + x2' + x3)" -> "(x1 | ~x2 | x3)")"""
        # Remove parentheses
        term = term.strip("()")
        
        if not term:
            return "1'b0"
        
        # Split by '+'
        literals = term.split(" + ")
        verilog_parts = []
        
        for lit in literals:
            lit = lit.strip()
            if lit.endswith("'"):
                # Complemented variable
                var = lit[:-1]
                verilog_parts.append(f"~{var}")
            else:
                verilog_parts.append(lit)
        
        return "(" + " | ".join(verilog_parts) + ")"

    def generate_html_report(self, filename="kmap_output.html", form='sop', module_name="logic_circuit"):
        """
        Generate a complete HTML file with:
        - Minimized expression display
        - Logic gate diagram (Graphviz DOT rendered to SVG via Viz.js)
        - Verilog code with syntax highlighting
        """
        terms, expression = self.minimize(form=form)
        verilog_code = self.generate_verilog(module_name=module_name, form=form)

        # Get Graphviz DOT (returned by _generate_logic_gates_mermaid)
        dot_source = self._generate_logic_gates_mermaid(terms, form=form)

        # Escape for JS template literal (keep backslashes for \n in DOT labels)
        dot_js = dot_source.replace("\\", "\\\\").replace("`", "\\`")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>K-Map Minimization Results</title>
<!-- Viz.js for Graphviz DOT -> SVG -->
<script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js"></script>
<script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5;
    }}
    .container {{
        background: white; border-radius: 8px; padding: 30px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
    }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #34495e; margin-top: 30px; }}
    .expression {{
        background: #ecf0f1; padding: 20px; border-radius: 5px;
        font-family: 'Courier New', monospace; font-size: 24px; text-align: center;
        color: #2c3e50; border-left: 4px solid #3498db;
    }}
    .logic-diagram {{ margin: 20px 0; padding: 20px; background: #fafafa; border-radius: 5px; overflow-x: auto; }}
    .verilog-code {{
        background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 5px;
        font-family: 'Courier New', monospace; font-size: 14px; overflow-x: auto; white-space: pre;
    }}
    .keyword {{ color: #3498db; }}
    .comment {{ color: #95a5a6; }}
    .wire {{ color: #e74c3c; }}
    .info {{
        background: #d4edda; border: 1px solid #c3e6cb; color: #155724;
        padding: 15px; border-radius: 5px; margin: 15px 0;
    }}
    .copy-btn {{
        background: #3498db; color: white; border: none; padding: 10px 20px;
        border-radius: 5px; cursor: pointer; font-size: 14px; margin-top: 10px;
    }}
    .copy-btn:hover {{ background: #2980b9; }}
</style>
</head>
<body>
<div class="container">
    <h1>K-Map Minimization Results</h1>

    <div class="info">
        <strong>Form:</strong> {form.upper()} ({'Sum of Products' if form.lower() == 'sop' else 'Product of Sums'})<br>
        <strong>Variables:</strong> {self.num_vars}<br>
        <strong>Terms:</strong> {len(terms)}<br>
        <strong>Convention:</strong> {self.convention}
    </div>

    <h2>Minimized Expression</h2>
    <div class="expression">
        F = {expression if expression else ('0' if form.lower() == 'sop' else '1')}
    </div>

    <h2>Logic Gate Diagram</h2>
    <div class="logic-diagram">
        <div id="graphviz"></div>
    </div>

    <h2>Verilog HDL Code</h2>
    <button class="copy-btn" onclick="copyVerilog()">Copy Verilog Code</button>
    <div class="verilog-code" id="verilog">{self._highlight_verilog(verilog_code)}</div>
</div>

<script>
    // Render DOT to SVG
    const dot = `{dot_js}`;
    const viz = new Viz();
    viz.renderSVGElement(dot)
      .then(svg => {{
        const container = document.getElementById('graphviz');
        container.innerHTML = '';
        container.appendChild(svg);
      }})
      .catch(err => {{
        const container = document.getElementById('graphviz');
        container.innerHTML = '<pre style="color:#c0392b;white-space:pre-wrap"></pre>';
        container.firstChild.textContent = 'Failed to render Graphviz diagram:\\n' + err;
      }});

    function copyVerilog() {{
        const code = `{verilog_code.replace('`', '\\`')}`;
        navigator.clipboard.writeText(code).then(() => {{
            alert('Verilog code copied to clipboard!');
        }});
    }}
</script>
</body>
</html>"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return filename

    def _highlight_verilog(self, code):
        """Apply basic syntax highlighting to Verilog code"""
        # Escape HTML
        code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Highlight keywords
        keywords = ['module', 'endmodule', 'input', 'output', 'wire', 'assign']
        for kw in keywords:
            code = code.replace(kw, f'<span class="keyword">{kw}</span>')
        
        # Highlight comments
        lines = code.split('\n')
        highlighted_lines = []
        for line in lines:
            if '//' in line:
                parts = line.split('//', 1)
                line = parts[0] + '<span class="comment">//' + parts[1] + '</span>'
            highlighted_lines.append(line)
        
        return '\n'.join(highlighted_lines)

    def _generate_logic_gates_mermaid(self, terms, form='sop'):
        """
        Generate Graphviz DOT language for logic circuit with actual gate symbols.
        
        Args:
            terms: List of minimized terms
            form: 'sop' or 'pos'
            
        Returns:
            String containing Graphviz DOT code
        """
        if not terms:
            # Constant output
            value = '0' if form.lower() == 'sop' else '1'
            dot = """digraph LogicCircuit {
        rankdir=LR;
        node [shape=circle, style=filled, fillcolor=lightblue];
        
        CONST [label=\"""" + value + """\" shape=box, fillcolor=lightgray];
        F [label="F", fillcolor=salmon];
        
        CONST -> F;
    }"""
            return dot
        
        if form.lower() == 'sop':
            return self._generate_sop_graphviz(terms)
        else:
            return self._generate_pos_graphviz(terms)

    def _generate_sop_graphviz(self, terms):
        """Generate Graphviz DOT for Sum of Products circuit"""
        dot = """digraph SOP_Circuit {
        rankdir=LR;
        node [fontname="Arial"];
        edge [arrowsize=0.8];
        
        // Graph attributes
        graph [splines=ortho, nodesep=0.8, ranksep=1.2];
        
    """
        
        # Collect all variables
        all_vars = set()
        for term in terms:
            vars_in_term = self._extract_variables(term)
            all_vars.update([var for var, _ in vars_in_term])
        
        sorted_vars = sorted(list(all_vars), key=lambda x: int(x[1:]))
        
        # Input nodes
        dot += "    // Input variables\n"
        dot += "    subgraph cluster_inputs {\n"
        dot += "        rank=same;\n"
        dot += "        style=invis;\n"
        for var in sorted_vars:
            dot += f'        {var} [label="{var}", shape=plaintext, fontsize=14];\n'
        dot += "    }\n\n"
        
        # NOT gates for complemented variables
        need_not = set()
        for term in terms:
            vars_in_term = self._extract_variables(term)
            for var, comp in vars_in_term:
                if comp:
                    need_not.add(var)
        
        if need_not:
            dot += "    // NOT gates\n"
            for var in sorted(need_not):
                dot += f'    NOT_{var} [label="NOT", shape=invtriangle, style=filled, fillcolor=lightyellow, width=0.6, height=0.6];\n'
                dot += f'    {var} -> NOT_{var};\n'
            dot += "\n"
        
        # AND gates for each term
        dot += "    // AND gates (product terms)\n"
        for i, term in enumerate(terms):
            vars_in_term = self._extract_variables(term)
            num_inputs = len(vars_in_term)
            
            # Clean term label for display
            term_label = term.replace("'", "̄")
            
            dot += f'    AND{i} [label="AND\\n{term_label}", shape=trapezium, style=filled, fillcolor=lightgreen, width=1.2, height=0.8, fontsize=10];\n'
            
            # Connect inputs to AND gate
            for var, comp in vars_in_term:
                if comp:
                    dot += f'    NOT_{var} -> AND{i};\n'
                else:
                    dot += f'    {var} -> AND{i};\n'
        
        dot += "\n"
        
        # OR gate (if multiple terms)
        if len(terms) > 1:
            dot += "    // OR gate (final sum)\n"
            dot += '    OR [label="OR", shape=trapezium, style=filled, fillcolor=lightcoral, width=1.0, height=0.8];\n'
            for i in range(len(terms)):
                dot += f'    AND{i} -> OR;\n'
            dot += '    OR -> F;\n\n'
        else:
            dot += '    AND0 -> F;\n\n'
        
        # Output node
        dot += "    // Output\n"
        dot += '    F [label="F", shape=doublecircle, style=filled, fillcolor=salmon, width=0.7, height=0.7];\n'
        
        dot += "}\n"
        return dot

    def _generate_pos_graphviz(self, terms):
        """Generate Graphviz DOT for Product of Sums circuit"""
        dot = """digraph POS_Circuit {
        rankdir=LR;
        node [fontname="Arial"];
        edge [arrowsize=0.8];
        
        // Graph attributes
        graph [splines=ortho, nodesep=0.8, ranksep=1.2];
        
    """
        
        # Collect all variables
        all_vars = set()
        for term in terms:
            vars_in_term = self._extract_variables(term)
            all_vars.update([var for var, _ in vars_in_term])
        
        sorted_vars = sorted(list(all_vars), key=lambda x: int(x[1:]))
        
        # Input nodes
        dot += "    // Input variables\n"
        dot += "    subgraph cluster_inputs {\n"
        dot += "        rank=same;\n"
        dot += "        style=invis;\n"
        for var in sorted_vars:
            dot += f'        {var} [label="{var}", shape=plaintext, fontsize=14];\n'
        dot += "    }\n\n"
        
        # NOT gates
        need_not = set()
        for term in terms:
            vars_in_term = self._extract_variables(term)
            for var, comp in vars_in_term:
                if comp:
                    need_not.add(var)
        
        if need_not:
            dot += "    // NOT gates\n"
            for var in sorted(need_not):
                dot += f'    NOT_{var} [label="NOT", shape=invtriangle, style=filled, fillcolor=lightyellow, width=0.6, height=0.6];\n'
                dot += f'    {var} -> NOT_{var};\n'
            dot += "\n"
        
        # OR gates for each term
        dot += "    // OR gates (sum terms)\n"
        for i, term in enumerate(terms):
            vars_in_term = self._extract_variables(term)
            
            # Clean term label
            term_label = term.replace("'", "̄").strip("()")
            
            dot += f'    OR{i} [label="OR\\n({term_label})", shape=trapezium, style=filled, fillcolor=lightcoral, width=1.2, height=0.8, fontsize=10];\n'
            
            # Connect inputs
            for var, comp in vars_in_term:
                if comp:
                    dot += f'    NOT_{var} -> OR{i};\n'
                else:
                    dot += f'    {var} -> OR{i};\n'
        
        dot += "\n"
        
        # AND gate (if multiple terms)
        if len(terms) > 1:
            dot += "    // AND gate (final product)\n"
            dot += '    AND [label="AND", shape=trapezium, style=filled, fillcolor=lightgreen, width=1.0, height=0.8];\n'
            for i in range(len(terms)):
                dot += f'    OR{i} -> AND;\n'
            dot += '    AND -> F;\n\n'
        else:
            dot += '    OR0 -> F;\n\n'
        
        # Output node
        dot += "    // Output\n"
        dot += '    F [label="F", shape=doublecircle, style=filled, fillcolor=salmon, width=0.7, height=0.7];\n'
        
        dot += "}\n"
        return dot

    def _extract_variables(self, term):
        """
        Extract variables and their complementation status from a term.
        Returns list of tuples: [(var_name, is_complemented), ...]
        """
        # Remove parentheses for POS terms
        term = term.strip("()")
        
        if " + " in term:
            # POS term - split by +
            literals = term.split(" + ")
            result = []
            for lit in literals:
                lit = lit.strip()
                if lit.endswith("'"):
                    result.append((lit[:-1], True))
                else:
                    result.append((lit, False))
            return result
        else:
            # SOP term
            result = []
            i = 0
            while i < len(term):
                if term[i] == 'x':
                    var_num = ""
                    i += 1
                    while i < len(term) and term[i].isdigit():
                        var_num += term[i]
                        i += 1
                    
                    var_name = f"x{var_num}"
                    
                    if i < len(term) and term[i] == "'":
                        result.append((var_name, True))
                        i += 1
                    else:
                        result.append((var_name, False))
                else:
                    i += 1
            return result

    # ---------- Display ---------- #
    def print_kmap(self):
        """Pretty print the K-map with headers."""
        print("     " + "  ".join(self.col_labels))
        for i, row in enumerate(self.kmap):
            print(f"{self.row_labels[i]}   " + "  ".join(str(val) for val in row))