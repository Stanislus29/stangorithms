"""
kmapsolver.py by Stan's Technologies, library to easily solve 2-4 variable kmaps.

A Karnaugh Map (K-map) solver that implements the Quine-McCluskey algorithm with
optimizations for 2-4 variable Boolean functions. Supports both Sum of Products (SOP)
and Product of Sums (POS) minimization with don't care conditions.
""" 

# Author: Somtochukwu Stanislus Emeka-Onwuneme 
# Publication Date: 8th September, 2025
# Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme
#---------------------------------------------------------

from collections import defaultdict

class KMapSolver:
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
        
        Args:
            form: 'sop' for Sum of Products or 'pos' for Product of Sums
            
        Returns:
            tuple: (list of minimized terms, complete expression string)
        """
        if form.lower() not in ['sop', 'pos']:
            raise ValueError("form must be either 'sop' or 'pos'")

        # For POS, we group 0's; for SOP, we group 1's
        target_val = 0 if form.lower() == 'pos' else 1

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
        prime_terms = []
        prime_covers = []
        for gmask in prime_groups:
            cover_mask = 0  # Tracks cells with target value covered by this group
            bits_list = []  # Collects binary representations for term generation
            
            # Process each cell in the group
            temp = gmask
            while temp:
                # Extract least significant 1-bit
                low = temp & -temp
                idx = low.bit_length() - 1
                temp -= low
                
                # Map bit position back to K-map coordinates
                r, c = self._index_to_rc[idx]
                
                # If cell has target value, include in coverage mask
                if self.kmap[r][c] == target_val:
                    cover_mask |= 1 << idx
                    
                # Always include cell bits for term generation
                bits_list.append(self._cell_bits[r][c])
                
            # Skip groups that don't cover any target cells
            if cover_mask == 0:
                continue
                
            # Generate Boolean term and save with its coverage
            term_str = simplify_method(bits_list)
            prime_terms.append(term_str)
            prime_covers.append(cover_mask)

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

        # Greedy set cover for remaining uncovered minterms
        remaining_mask = all_minterms_mask & ~covered_mask
        selected = set(essential_indices)
        
        while remaining_mask:
            # Find prime implicant covering most uncovered minterms
            best_idx, best_cover_count = None, -1
            for idx in range(len(prime_covers)):
                if idx in selected:
                    continue
                # Count bits covered by this prime
                cover = prime_covers[idx] & remaining_mask
                count = cover.bit_count()
                if count > best_cover_count:
                    best_cover_count = count
                    best_idx = idx
                    
            # Stop if no improvement possible
            if best_idx is None or best_cover_count == 0:
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
        return final_terms, join_operator.join(final_terms)

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
    
    # ---------- Display ---------- #
    def print_kmap(self):
        """Pretty print the K-map with headers."""
        print("     " + "  ".join(self.col_labels))
        for i, row in enumerate(self.kmap):
            print(f"{self.row_labels[i]}   " + "  ".join(str(val) for val in row))