"""
lib_kmap_solver.py, library to easily solve 2-4 variable kmaps.

Developed from kmapsolver.py, kmap_solver_prototype.py
""" 

# Author: Somtochukwu Stanislus Emeka-Onwuneme 
# Publication Date: 8th September, 2025
# Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme
#---------------------------------------------------------

from collections import defaultdict

class KMapSolver:
    def __init__(self, kmap, convention = "vranseic"):
        """
        Initialize with a given K-map.
        kmap: 2D list with values 0, 1, or 'd'
        """
        self.kmap = kmap
        self.num_rows = len(kmap)
        self.num_cols = len(kmap[0])
        self.convention = convention.lower().strip()

        # Determine number of variables (2, 3, or 4)
        size = self.num_rows * self.num_cols
        if size == 4:
            self.num_vars = 2
            self.row_labels = ["0", "1"]
            self.col_labels = ["0", "1"]
        elif size == 8:
            self.num_vars = 3
            self.row_labels = ["0", "1"]
            self.col_labels = ["00", "01", "11", "10"]
        elif size == 16:
            self.num_vars = 4
            self.row_labels = ["00", "01", "11", "10"]
            self.col_labels = ["00", "01", "11", "10"]
        else:
            raise ValueError("K-map must be 2x2 (2 vars), 2x4/4x2 (3 vars), or 4x4 (4 vars)")

        # Group sizes (powers of 2 only)
        self.group_sizes = [(1,1),(1,2),(2,1),(2,2),
                            (1,4),(4,1),(2,4),(4,2),(4,4)]

    # ---------- Utility functions ---------- #
    def _cell_to_term(self, r, c):
        """Convert (row, col) to a minterm index."""
        bits = self.col_labels[c] + self.row_labels[r]
        return int(bits, 2)
    
    # def _cell_to_maxterm(self, r, c):
    #     """Convert (row, col) to a maxterm index."""
    #     bits = self.col_labels[c] + self.row_labels[r]
    #     return int(bits, 2)

    def _cell_to_bits(self, r, c):
        """Return the bit string representing (row, col) based on the convention."""
        if self.convention == "mano_kime":
            # Mano-Kime → rows = x1x2, cols = x3x4
            bits = self.row_labels[r] + self.col_labels[c]
        else:
            # Vranesic → rows = x3x4, cols = x1x2
            bits = self.col_labels[c] + self.row_labels[r]
        return bits


    def _get_group_coords(self, r, c, h, w):
        """Get wrapped coordinates for a group starting at (r,c)."""
        return [((r+i) % self.num_rows, (c+j) % self.num_cols)
                for i in range(h) for j in range(w)]

    # ---------- Group finding ---------- #
    def find_all_groups(self, allow_dontcare=False):
        """Find all groups of 1’s (and optionally 'd')."""
        groups = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                for h, w in self.group_sizes:
                    if h > self.num_rows or w > self.num_cols:
                        continue

                    coords = self._get_group_coords(r, c, h, w)

                    if allow_dontcare:
                        # ✅ allow 'd' only if at least one '1' is inside
                        if all(self.kmap[rr][cc] in (1, 'd') for rr, cc in coords) and \
                           any(self.kmap[rr][cc] == 1 for rr, cc in coords):
                            groups.append(frozenset(coords))
                    else:
                        if all(self.kmap[rr][cc] == 1 for rr, cc in coords):
                            groups.append(frozenset(coords))
        return list(set(groups))

    def find_all_groups_pos(self, allow_dontcare=False):
        """Find all groups of 0's (and optionally 'd') for POS."""
        groups = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                for h, w in self.group_sizes:
                    if h > self.num_rows or w > self.num_cols:
                        continue

                    coords = self._get_group_coords(r, c, h, w)

                    if allow_dontcare:
                        if all(self.kmap[rr][cc] in (0, 'd') for rr, cc in coords) and \
                        any(self.kmap[rr][cc] == 0 for rr, cc in coords):
                            groups.append(frozenset(coords))
                    else:
                        if all(self.kmap[rr][cc] == 0 for rr, cc in coords):
                            groups.append(frozenset(coords))
        return list(set(groups))

    def filter_prime_implicants(self, groups):
        """Remove groups that are subsets of others."""
        primes = []
        for g in groups:
            if not any((g < other) for other in groups if other != g):
                primes.append(g)
        return primes

    # ---------- Boolean simplification ---------- #
    def _simplify_group_bits(self, bits_list):
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
        Minimize the K-map expression.
        Args:
            form: 'sop' for Sum of Products or 'pos' for Product of Sums
        Returns:
            tuple: (list of terms, string expression)
        """
        if form.lower() not in ['sop', 'pos']:
            raise ValueError("form must be either 'sop' or 'pos'")

        # Choose appropriate methods based on form
        if form.lower() == 'pos':
            groups = self.find_all_groups_pos(allow_dontcare=True)
            simplify_method = self._simplify_group_bits_pos
            cell_to_term = self._cell_to_term
            join_operator = " * "
        else:  # SOP
            groups = self.find_all_groups(allow_dontcare=True)
            simplify_method = self._simplify_group_bits
            cell_to_term = self._cell_to_term
            join_operator = " + "

        # Step 2: reduce to prime implicants
        prime_groups = self.filter_prime_implicants(groups)

        # Step 3: compute terms and covers
        prime_terms, prime_covers = [], []
        for g in prime_groups:
            coords = sorted(g)
            terms = sorted(cell_to_term(r, c) for r, c in coords
                        if self.kmap[r][c] == (0 if form.lower() == 'pos' else 1))
            if not terms:
                continue
            bits_list = [self._cell_to_bits(r, c) for r, c in coords]
            term_str = simplify_method(bits_list)
            prime_terms.append(term_str)
            prime_covers.append(terms)

        # Step 4: build prime implicant chart
        minterm_to_primes = defaultdict(list)
        all_minterms = set()
        for idx, cover in enumerate(prime_covers):
            for m in cover:
                minterm_to_primes[m].append(idx)
                all_minterms.add(m)

        # Step 5: essential primes
        essential_indices = set()
        for m, primes in minterm_to_primes.items():
            if len(primes) == 1:
                essential_indices.add(primes[0])

        essential_terms = [prime_terms[i] for i in sorted(essential_indices)]

        # Step 6: mark covered
        covered_minterms = set()
        for i in essential_indices:
            covered_minterms.update(prime_covers[i])

        # Step 7: greedy covering
        remaining = set(all_minterms) - covered_minterms
        selected = set(essential_indices)
        while remaining:
            best_idx, best_cover_count = None, -1
            for idx in range(len(prime_groups)):
                if idx in selected: 
                    continue
                cover = set(prime_covers[idx]) & remaining
                if len(cover) > best_cover_count:
                    best_cover_count = len(cover)
                    best_idx = idx
            if best_idx is None:
                break
            selected.add(best_idx)
            covered_minterms.update(prime_covers[best_idx])
            remaining = set(all_minterms) - covered_minterms

        # Step 8: remove redundant terms
        def covers_with_terms(indices):
            covered = set()
            for idx in indices:
                covered.update(prime_covers[idx])
            return covered

        chosen = set(sorted(selected))
        for idx in sorted(list(chosen)):
            trial = chosen - {idx}
            if covers_with_terms(trial) == covers_with_terms(chosen):
                chosen = trial

        final_terms = [prime_terms[i] for i in sorted(chosen)]
        return final_terms, join_operator.join(final_terms)

    # ---------- Display ---------- #
    def print_kmap(self):
        """Pretty print the K-map with headers."""
        print("     " + "  ".join(self.col_labels))
        for i, row in enumerate(self.kmap):
            print(f"{self.row_labels[i]}   " + "  ".join(str(val) for val in row))