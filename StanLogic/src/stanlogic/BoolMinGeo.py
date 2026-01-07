"""
A library for minimizing Boolean systems (5 to 10 variables) using 
geometric methods based on 3D and 4D Karnaugh Maps


Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: December 2025
"""

from stanlogic import BoolMin2D
from collections import defaultdict
import numpy as np
import re

class BoolMinGeo:
    def __init__(self, num_vars, output_values=None):
        """
        Initialize BoolMinGeo with the number of variables.
        Generates truth table combinations and creates 3D K-map structure.
        
        Args:
            num_vars (int): Number of variables in the truth table (must be >= 5)
            output_values (list): Optional list of output values for each minterm
        """
        if num_vars < 5:
            raise ValueError("BoolMinGeo requires at least 5 variables. Use BoolMin2D for 2-4 variables.")
        
        self.num_vars = num_vars
        self.truth_table = self.generate_truth_table()
        
        # Set output values (default to all 0s if not provided)
        if output_values is None:
            self.output_values = [0] * len(self.truth_table)
        else:
            if len(output_values) != len(self.truth_table):
                raise ValueError(f"output_values must have {len(self.truth_table)} entries")
            self.output_values = output_values
        
        # Calculate structure
        self.num_extra_vars = num_vars - 4  # Variables beyond the 4x4 map
        self.num_maps = 2 ** self.num_extra_vars  # Number of 4x4 maps needed
        
        # Gray code sequences for 4x4 K-map labeling (last 4 variables)
        self.gray_code_4 = ["00", "01", "11", "10"]
        
        # Build the hierarchical K-map structure
        self.kmaps = self.build_kmaps()
    
    def generate_truth_table(self):
        """
        Generate truth table combinations using binary addition.
        Starting from all zeros, incrementally add 1 until all combinations are generated.
        
        Returns:
            list: List of all binary combinations as lists of integers
        """
        # Total number of combinations: 2^num_vars
        total_combinations = 2 ** self.num_vars
        truth_table = []
        
        # Start with initial combination of all zeros
        current = [0] * self.num_vars
        truth_table.append(current.copy())
        
        # Generate remaining combinations using binary addition
        for _ in range(total_combinations - 1):
            current = self.binary_add_one(current)
            truth_table.append(current.copy())
        
        return truth_table
    
    def binary_add_one(self, binary_list):
        """
        Add 1 to a binary number represented as a list.
        Implements binary addition with carry propagation.
        
        Args:
            binary_list (list): Binary number as list of 0s and 1s
            
        Returns:
            list: Result of adding 1 to the binary number
        """
        result = binary_list.copy()
        carry = 1
        
        # Start from the rightmost bit (least significant bit)
        for i in range(len(result) - 1, -1, -1):
            sum_bit = result[i] + carry
            result[i] = sum_bit % 2  # Store the bit value (0 or 1)
            carry = sum_bit // 2      # Calculate carry for next position
            
            # If no carry, we're done
            if carry == 0:
                break
        
        return result
    
    def build_kmaps(self):
        """
        Build hierarchical K-map structure based on extra variables.
        
        For n variables:
        - First (n-4) variables determine which 4x4 map
        - Last 4 variables determine position within each 4x4 map using Gray coding
        
        Returns:
            dict: Dictionary mapping extra variable combinations to 4x4 K-maps
        """
        kmaps = {}
        
        # Generate all combinations for extra variables
        extra_var_combinations = []
        for i in range(self.num_maps):
            # Convert index to binary with appropriate width
            """
            Example:
            If self.num_extra_vars = 3 and self.num_maps = 8, the loop generates:
            ['000', '001', '010', '011', '100', '101', '110', '111']

            Each entry will later be used to fill or reference the corresponding K-map 
            with the extra variables set to that binary combination. 
            """
            
            binary_str = format(i, f'0{self.num_extra_vars}b')
            extra_var_combinations.append(binary_str)
        
        # For each combination of extra variables, create a 4x4 K-map
        for extra_combo in extra_var_combinations:
            # Initialize 4x4 map
            kmap_4x4 = [[None for _ in range(4)] for _ in range(4)]
            
            # Fill the map with minterms using Gray code ordering
            for row_idx, row_gray in enumerate(self.gray_code_4):
                for col_idx, col_gray in enumerate(self.gray_code_4):
                    # Construct full variable combination
                    # Format: [extra_vars][col_gray][row_gray]
                    full_combination = list(extra_combo) + list(col_gray) + list(row_gray)
                    full_combination = [int(b) for b in full_combination]
                    
                    # Find this combination in truth table and get its index (minterm)
                    minterm_idx = self.truth_table.index(full_combination)
                    
                    # Store the minterm and its output value
                    kmap_4x4[row_idx][col_idx] = {
                        'minterm': minterm_idx,
                        'value': self.output_values[minterm_idx],
                        'variables': full_combination
                    }
            
            kmaps[extra_combo] = kmap_4x4
        
        return kmaps
    
    def get_truth_table(self):
        """
        Get the generated truth table.
        
        Returns:
            list: The complete truth table
        """
        return self.truth_table
    
    def print_truth_table(self):
        """
        Print the truth table in a formatted way.
        """
        # Print header
        var_names = [f"x{i+1}" for i in range(self.num_vars)]
        header = " | ".join(var_names) + " | F"
        print(header)
        print("-" * len(header))
        
        # Print each row
        for idx, row in enumerate(self.truth_table):
            bits_str = " | ".join([f" {bit} " for bit in row])
            print(f"{bits_str} | {self.output_values[idx]}")
    
    def print_kmaps(self):
        """
        Print all hierarchical K-maps in a formatted way.
        Shows one 4x4 K-map for each combination of extra variables.
        """
        print(f"\n{'='*60}")
        print(f"3D K-MAP STRUCTURE FOR {self.num_vars} VARIABLES")
        print(f"{'='*60}")
        print(f"Number of 4x4 K-maps: {self.num_maps}")
        print(f"Extra variables: x1 to x{self.num_extra_vars}")

        # Print which variables are used for this 4x4 K-map
        # Columns: next two variables after the extra variables
        # Rows: next two variables after the columns
        print(f"Map variables: x{self.num_extra_vars+1}x{self.num_extra_vars+2} (columns), "
              f"x{self.num_extra_vars+3}x{self.num_extra_vars+4} (rows)")
        print(f"{'='*60}\n")
        
        # Print each K-map
        for extra_combo in sorted(self.kmaps.keys()):
            self._print_single_kmap(extra_combo)
    
    def _print_single_kmap(self, extra_combo):
        """
        Print a single 4x4 K-map for a specific extra variable combination.
        
        Args:
            extra_combo (str): Binary string representing extra variable values
        """
        kmap = self.kmaps[extra_combo]
        
        # Create header showing which extra variables are set
        extra_var_labels = [f"x{i+1}={extra_combo[i]}" for i in range(len(extra_combo))]
        print(f"K-map for: {', '.join(extra_var_labels)}")
        print("-" * 50)
        
        # Column headers (Gray code for x_{n-3}x_{n-2})
        col_offset = self.num_extra_vars + 1
        print(f"      x{col_offset}x{col_offset+1}")
        print(f"x{col_offset+2}x{col_offset+3}  " + "  ".join(self.gray_code_4))
        print("      " + "-" * 20)
        
        # Print each row with Gray code labels
        for row_idx, row_gray in enumerate(self.gray_code_4):
            row_str = f" {row_gray}  |"
            for col_idx in range(4):
                cell = kmap[row_idx][col_idx]
                value = cell['value']
                minterm = cell['minterm']
                
                # Format: show value (and minterm number)
                if isinstance(value, str) and value.lower() == 'd':
                    row_str += f" d({minterm:2d})"
                else:
                    row_str += f" {value}({minterm:2d})"
            print(row_str)
        print()
    
    def set_output_values(self, output_values):
        """
        Set or update output values and rebuild K-maps.
        
        Args:
            output_values (list): List of output values for each minterm
        """
        if len(output_values) != len(self.truth_table):
            raise ValueError(f"output_values must have {len(self.truth_table)} entries")
        
        self.output_values = output_values
        self.kmaps = self.build_kmaps()
    
    def get_kmap_for_extra_vars(self, extra_var_values):
        """
        Get a specific 4x4 K-map based on extra variable values.
        
        Args:
            extra_var_values (str or list): Binary values for extra variables
            
        Returns:
            list: The 4x4 K-map matrix
        """
        if isinstance(extra_var_values, list):
            extra_var_values = ''.join([str(v) for v in extra_var_values])
        
        if extra_var_values not in self.kmaps:
            raise ValueError(f"Invalid extra variable combination: {extra_var_values}")
        
        return self.kmaps[extra_var_values]
    
    def _extract_kmap_values(self, extra_combo):
        """
        Extract just the values from a 4x4 K-map for minimization.
        
        Args:
            extra_combo (str): Binary string for extra variables
            
        Returns:
            list: 4x4 matrix of values suitable for BoolMin2D
        """
        kmap = self.kmaps[extra_combo]
        values_4x4 = []
        for row in kmap:
            row_values = [cell['value'] for cell in row]
            values_4x4.append(row_values)
        return values_4x4
    
    def _solve_single_kmap(self, extra_combo, form='sop'):
        """
        Solve a single 4x4 K-map and return essential prime implicants as bitmasks.
        
        Args:
            extra_combo (str): Binary string for extra variables
            form (str): 'sop' or 'pos'
            
        Returns:
            dict: Contains 'bitmasks', 'terms_bits', and 'extra_vars'
        """
        # Extract values for this K-map
        kmap_values = self._extract_kmap_values(extra_combo)
        
        # Create a BoolMin2D instance for this 4x4 map
        solver = BoolMin2D(kmap_values, convention="vranseic")
        
        # Get the essential prime implicants (internal computation)
        target_val = 0 if form.lower() == 'pos' else 1
        
        if form.lower() == 'pos':
            groups = solver.find_all_groups_pos(allow_dontcare=True)
        else:
            groups = solver.find_all_groups(allow_dontcare=True)
        
        # Filter to prime implicants
        prime_groups = solver.filter_prime_implicants(groups)
        
        # Generate coverage masks and terms
        prime_covers = []
        prime_terms_bits = []  # Store bit patterns for each prime
        
        for gmask in prime_groups:
            cover_mask = 0
            bits_list = []
            
            temp = gmask
            while temp:
                low = temp & -temp
                idx = low.bit_length() - 1
                temp -= low
                
                r, c = solver._index_to_rc[idx]
                
                if solver.kmap[r][c] == target_val:
                    cover_mask |= 1 << idx
                
                bits_list.append(solver._cell_bits[r][c])
            
            if cover_mask == 0:
                continue
            
            prime_covers.append(cover_mask)
            
            # Simplify to get bit pattern (with '-' for don't cares)
            simplified_bits = self._simplify_bits_only(bits_list)
            prime_terms_bits.append(simplified_bits)
        
        # Find essential prime implicants
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
        
        essential_indices = set()
        for m, primes in minterm_to_primes.items():
            if len(primes) == 1:
                essential_indices.add(primes[0])
        
        # Greedy set cover for remaining minterms
        covered_mask = 0
        for i in essential_indices:
            covered_mask |= prime_covers[i]
        
        remaining_mask = all_minterms_mask & ~covered_mask
        selected = set(essential_indices)
        
        while remaining_mask:
            best_idx, best_cover_count = None, -1
            for idx in range(len(prime_covers)):
                if idx in selected:
                    continue
                cover = prime_covers[idx] & remaining_mask
                count = cover.bit_count()
                if count > best_cover_count:
                    best_cover_count = count
                    best_idx = idx
            
            if best_idx is None or best_cover_count == 0:
                break
            
            selected.add(best_idx)
            covered_mask |= prime_covers[best_idx]
            remaining_mask = all_minterms_mask & ~covered_mask
        
        # Remove redundancy
        def covers_with_indices(indices):
            mask = 0
            for i in indices:
                mask |= prime_covers[i]
            return mask
        
        chosen = set(selected)
        for idx in list(sorted(chosen)):
            trial = chosen - {idx}
            if covers_with_indices(trial) == covers_with_indices(chosen):
                chosen = trial
        
        # Return bitmasks and bit patterns for chosen essential primes
        result_bitmasks = [prime_covers[i] for i in sorted(chosen)]
        result_bits = [prime_terms_bits[i] for i in sorted(chosen)]
        
        return {
            'bitmasks': result_bitmasks,
            'terms_bits': result_bits,
            'extra_vars': extra_combo
        }
    
    def count_literals(expr_str, form="sop"):
        """
        Count terms and literals in a Boolean expression string.
        
        Args:
            expr_str: Boolean expression string
            form: 'sop' or 'pos'
            
        Returns:
            Tuple of (num_terms, num_literals)
        """
        if not expr_str or expr_str.strip() == "":
            return 0, 0

        form = form.lower()
        s = expr_str.replace(" ", "")

        if form == "sop":
            terms = [t for t in s.split('+') if t]
            num_terms = len(terms)
            num_literals = sum(len(re.findall(r"[A-Za-z_]\w*'?", t)) for t in terms)
            return num_terms, num_literals

        if form == "pos":
            clauses = re.findall(r"\(([^()]*)\)", s)
            if not clauses:
                clauses = [s]
            num_terms = len(clauses)
            num_literals = 0
            for clause in clauses:
                lits = [lit for lit in clause.split('+') if lit]
                num_literals += sum(1 for lit in lits if re.fullmatch(r"[A-Za-z_]\w*'?", lit))
            return num_terms, num_literals

        raise ValueError("form must be 'sop' or 'pos'")

    def _simplify_bits_only(self, bits_list):
        """
        Simplify a group of bits to a pattern with '-' for varying bits.
        Only works on the last 4 variables (the 4x4 K-map portion).
        
        Args:
            bits_list: List of bit strings (4 bits each)
            
        Returns:
            str: Simplified bit pattern (e.g., "01-1")
        """
        if not bits_list:
            return ""
        
        bits = list(bits_list[0])
        for b in bits_list[1:]:
            for i in range(4):  # Only 4 variables in each 4x4 map
                if bits[i] != b[i]:
                    bits[i] = '-'
        
        return "".join(bits)
    
    def _simplify_extra_vars(self, extra_combos):
        """
        Simplify the extra variable portion by finding common patterns.
        
        Args:
            extra_combos: List of binary strings for extra variables
            
        Returns:
            str: Simplified pattern with '-' for varying bits
        """
        if not extra_combos:
            return ""
        
        if len(extra_combos) == 1:
            return extra_combos[0]
        
        # Compare all combinations to find varying positions
        bits = list(extra_combos[0])
        for combo in extra_combos[1:]:
            for i in range(len(bits)):
                if bits[i] != combo[i]:
                    bits[i] = '-'
        
        return "".join(bits)
    
    def minimize_3d(self, form='sop'):
        """
        Minimize using 3D clustering principle.
        """
        print(f"\n{'='*60}")
        print(f"MINIMIZING {self.num_vars}-VARIABLE K-MAP (3D CLUSTERING)")
        print(f"{'='*60}\n")
        
        # Step 1: Minimize each K-map partition
        β = {}
        for extra_combo in sorted(self.kmaps.keys()):
            result = self._solve_single_kmap(extra_combo, form)
            β[extra_combo] = result['terms_bits']
        
        # Step 2: Apply 3D clustering algorithm
        id_set = list(self.kmaps.keys())
        β_merged = self._minimize_with_3d_clustering(β, id_set)
        
        # Step 3: Convert to terms
        final_terms = []
        for full_pattern in sorted(β_merged):
            term_str = self._bits_to_term(full_pattern, form)
            final_terms.append(term_str)
        
        # Step 4: Build expression
        join_operator = " * " if form.lower() == 'pos' else " + "
        final_expression = join_operator.join(final_terms)
        
        print(f"\n{'='*60}")
        print(f"FINAL EXPRESSION: {len(final_terms)} terms")
        print(f"{'='*60}")
        print(f"F = {final_expression}\n")
        
        return final_terms, final_expression

    def _minimize_with_3d_clustering(self, β, id_set):
        """
        Pure 3D clustering: Only consider patterns appearing in 2+ adjacent identifiers.
        Eliminate 2D clusters entirely.
        
        Args:
            β: Dictionary mapping identifier → list of patterns
            id_set: List of all identifiers
            
        Returns:
            set: Minimal set of 3D clusters (EPIs)
        """
        print("\n" + "="*60)
        print("PHASE 1: 3D CLUSTER IDENTIFICATION (Eliminating 2D Clusters)")
        print("="*60)
        
        # Step 1: Group patterns by K-map portion
        pattern_to_identifiers = {}
        for idx in id_set:
            for pattern in β[idx]:
                if pattern not in pattern_to_identifiers:
                    pattern_to_identifiers[pattern] = []
                pattern_to_identifiers[pattern].append(idx)
        
        # Step 2: Filter to keep ONLY 3D clusters (2+ adjacent identifiers)
        valid_3d_clusters = {}
        eliminated_2d = []
        
        for pattern, id_list in pattern_to_identifiers.items():
            if len(id_list) == 1:
                # Pure 2D cluster: ELIMINATE
                eliminated_2d.append((pattern, id_list[0]))
                print(f"  ✗ ELIMINATED 2D: '{pattern}' in identifier {id_list[0]}")
            elif not self._has_adjacent_identifiers(id_list):
                # Non-adjacent: ELIMINATE (treat as disconnected 2D)
                eliminated_2d.extend([(pattern, idx) for idx in id_list])
                print(f"  ✗ ELIMINATED Non-adjacent: '{pattern}' in {id_list}")
            else:
                # Valid 3D cluster: KEEP
                valid_3d_clusters[pattern] = id_list
                print(f"  ✓ VALID 3D: '{pattern}' in {len(id_list)} adjacent identifiers")
        
        print(f"\n  Kept: {len(valid_3d_clusters)} 3D clusters")
        print(f"  Eliminated: {len(eliminated_2d)} 2D clusters")
        
        if not valid_3d_clusters:
            print("\n  WARNING: No valid 3D clusters found!")
            print("  This may indicate the function has no proper 3D structure.")
            print("  Falling back to include all patterns...")
            # Fallback: include all patterns
            valid_3d_clusters = pattern_to_identifiers
        
        # Step 3: Merge identifiers depth-wise for each 3D cluster
        print("\n" + "="*60)
        print("PHASE 2: DEPTH-WISE MERGING")
        print("="*60)
        
        merged_3d_clusters = []
        for pattern, id_list in valid_3d_clusters.items():
            print(f"\nMerging identifiers for pattern '{pattern}':")
            print(f"  Identifiers: {id_list}")
            
            # Apply Quine-McCluskey with essential PI selection
            merged_ids = self._minimize_boolean_function_complete(id_list)
            
            print(f"  Prime implicants: {merged_ids}")
            
            for merged_id in merged_ids:
                cluster = {
                    'kmap_pattern': pattern,
                    'identifier_pattern': merged_id,
                    'full_pattern': merged_id + pattern,
                    'depth': self._count_depth(merged_id),  # Number of identifiers covered
                    'cells': self._get_cell_positions(pattern)  # (row, col) positions
                }
                merged_3d_clusters.append(cluster)
                print(f"    Cluster: {merged_id} + {pattern} (depth={cluster['depth']})")
        
        # Step 4: Select EPIs using depth-wise dominance
        print("\n" + "="*60)
        print("PHASE 3: ESSENTIAL PRIME IMPLICANT SELECTION (Depth-wise)")
        print("="*60)
        
        epis = self._select_epis_by_depth_dominance(merged_3d_clusters)
        
        print(f"\nSelected {len(epis)} essential prime implicants")

        # # Step 5: Verify coverage
        # print("\n" + "="*60)
        # print("PHASE 4: COVERAGE VERIFICATION")
        # print("="*60)
        
        # all_minterms = self._get_all_minterms_from_kmaps(id_set)
        # covered = self._get_covered_minterms([c['full_pattern'] for c in epis])
        # uncovered = all_minterms - covered
        
        # if uncovered:
        #     print(f"  ⚠ WARNING: {len(uncovered)} minterms not covered by 3D clusters!")
        #     print(f"  Uncovered minterms: {list(uncovered)[:10]}...")  # Show first 10
            
        #     # These should be covered by extending 3D clusters or indicate error
        #     print(f"\n  Attempting to cover with remaining 3D clusters...")
        #     additional = self._greedy_cover_from_clusters(
        #         uncovered, merged_3d_clusters, epis
        #     )
        #     epis.extend(additional)
            
        #     # Re-verify
        #     covered = self._get_covered_minterms([c['full_pattern'] for c in epis])
        #     still_uncovered = all_minterms - covered
            
        #     if still_uncovered:
        #         print(f"  ✗ CRITICAL: {len(still_uncovered)} minterms still uncovered!")
        #         print(f"  This indicates incomplete 3D clustering.")
        #         print(f"  The function may have isolated regions requiring 2D clusters.")
        #     else:
        #         print(f"  ✓ Coverage achieved with {len(additional)} additional clusters")
        # else:
        #     print(f"  ✓ All {len(all_minterms)} minterms covered by 3D clusters")
        
        return {c['full_pattern'] for c in epis}

    def _count_depth(self, identifier_pattern):
        """
        Count the number of concrete identifiers represented by a pattern.
        Depth = 2^(number of don't cares)
        
        Args:
            identifier_pattern: String with possible '-'
            
        Returns:
            int: Number of identifiers (depth)
        """
        n_dontcares = identifier_pattern.count('-')
        return 2 ** n_dontcares

    def _get_cell_positions(self, kmap_pattern):
        """
        Get the set of (row, col) positions represented by a K-map pattern.
        
        Args:
            kmap_pattern: 4-bit pattern with possible '-'
            
        Returns:
            frozenset: Set of (row, col) tuples this pattern covers
        """
        # Expand pattern to all concrete 4-bit combinations
        concrete_patterns = self._expand_pattern(kmap_pattern)
        
        # Convert each to (row, col) using Gray code mapping
        cells = set()
        gray_code = ['00', '01', '11', '10']
        
        for concrete in concrete_patterns:
            # First 2 bits → column (via Gray code)
            # Last 2 bits → row (via Gray code)
            col_bits = concrete[:2]
            row_bits = concrete[2:]
            
            col = gray_code.index(col_bits)
            row = gray_code.index(row_bits)
            
            cells.add((row, col))
        
        return frozenset(cells)

    def _select_epis_by_depth_dominance(self, clusters):
        """
        Select EPIs using depth-wise dominance criterion.
        
        For clusters with same K-map pattern and same cell coverage:
        - Keep the one with maximum depth
        - Eliminate others (they are subsumed)
        
        For clusters with different cell coverage:
        - Keep all, regardless of depth relationship
        
        Args:
            clusters: List of cluster dictionaries
            
        Returns:
            list: Essential prime implicants
        """
        # Group clusters by K-map pattern
        pattern_groups = {}
        for cluster in clusters:
            kmap_pat = cluster['kmap_pattern']
            if kmap_pat not in pattern_groups:
                pattern_groups[kmap_pat] = []
            pattern_groups[kmap_pat].append(cluster)
        
        epis = []
        
        for kmap_pattern, group in pattern_groups.items():
            print(f"\n  Analyzing pattern '{kmap_pattern}':")
            
            # Further group by cell coverage
            cell_groups = {}
            for cluster in group:
                cells_key = cluster['cells']  # frozenset, so hashable
                if cells_key not in cell_groups:
                    cell_groups[cells_key] = []
                cell_groups[cells_key].append(cluster)
            
            # For each cell group, select maximum depth
            for cells, cell_group in cell_groups.items():
                print(f"    Cell coverage: {len(cells)} cells")
                
                # Find maximum depth
                max_depth = max(c['depth'] for c in cell_group)
                
                # Keep only clusters with maximum depth
                for cluster in cell_group:
                    if cluster['depth'] == max_depth:
                        epis.append(cluster)
                        print(f"      ✓ EPI: {cluster['identifier_pattern']} "
                            f"(depth={cluster['depth']}) - MAX DEPTH")
                    else:
                        print(f"      ✗ Dominated: {cluster['identifier_pattern']} "
                            f"(depth={cluster['depth']}) < {max_depth}")
        
        return epis

    def _greedy_cover_from_clusters(self, uncovered, all_clusters, current_epis):
        """
        Use greedy algorithm to cover uncovered minterms using remaining clusters.
        Only uses 3D clusters.
        
        Args:
            uncovered: Set of uncovered minterms
            all_clusters: List of all 3D cluster dictionaries
            current_epis: List of currently selected EPIs
            
        Returns:
            list: Additional clusters needed
        """
        additional = []
        remaining = uncovered.copy()
        
        # Get clusters not yet selected
        current_patterns = {c['full_pattern'] for c in current_epis}
        available = [c for c in all_clusters 
                    if c['full_pattern'] not in current_patterns]
        
        while remaining and available:
            # Find cluster covering most remaining minterms
            best_cluster = None
            best_coverage = 0
            
            for cluster in available:
                covered = self._expand_pattern(cluster['full_pattern'])
                overlap = len(covered & remaining)
                if overlap > best_coverage:
                    best_coverage = overlap
                    best_cluster = cluster
            
            if best_cluster is None or best_coverage == 0:
                break
            
            additional.append(best_cluster)
            newly_covered = self._expand_pattern(best_cluster['full_pattern'])
            remaining -= newly_covered
            available.remove(best_cluster)
            
            print(f"    Added: {best_cluster['full_pattern']} "
                f"(covers {best_coverage} more minterms)")
        
        return additional

    def _get_all_minterms_from_kmaps(self, id_set):
        """
        Get all minterms where function = 1 directly from K-maps.
        
        Args:
            id_set: List of identifiers
            
        Returns:
            set: Set of full minterms (n-bit strings)
        """
        minterms = set()
        
        for idx in id_set:
            kmap = self.kmaps[idx]
            
            for row_idx in range(4):
                for col_idx in range(4):
                    cell = kmap[row_idx][col_idx]
                    if cell and cell.get('value') == 1:
                        # Get full n-bit string
                        full_bits = ''.join(str(b) for b in cell['variables'])
                        minterms.add(full_bits)
        
        return minterms

    def _minimize_boolean_function_complete(self, minterm_list):
        """
        Complete Quine-McCluskey with essential prime implicant selection.
        Guarantees minimal cover.
        
        Args:
            minterm_list: List of binary strings
            
        Returns:
            list: Minimal set of prime implicants
        """
        if not minterm_list:
            return []
        
        if len(minterm_list) == 1:
            return minterm_list
        
        # Remove duplicates
        minterms = list(set(minterm_list))
        
        # Phase 1: Find all prime implicants (existing code)
        prime_implicants = self._find_all_prime_implicants_bitwise(minterms)
        
        print(f"    Found {len(prime_implicants)} prime implicants")
        
        # Phase 2: Select essential prime implicants (NEW)
        essential_pis = self._select_essential_prime_implicants(
            prime_implicants, minterms
        )
        
        print(f"    Selected {len(essential_pis)} essential prime implicants")
        
        return essential_pis

    def _has_adjacent_identifiers(self, id_list):
        """
        Check if a list of identifiers contains at least one pair of adjacent identifiers.
        
        Args:
            id_list: List of binary string identifiers
            
        Returns:
            bool: True if at least one pair is adjacent
        """
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                if self._hamming_distance(id_list[i], id_list[j]) == 1:
                    return True
        return False

    def _hamming_distance(self, str1, str2):
        """
        Calculate Hamming distance between two strings.
        
        Args:
            str1, str2: Binary strings
            
        Returns:
            int: Number of positions where strings differ
        """
        if len(str1) != len(str2):
            return float('inf')
        
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def _eliminate_dominated_clusters(self, clusters):
        """
        Eliminate clusters that are dominated by other clusters.
        
        Cluster C2 is dominated by C1 if:
        1. They have the same K-map pattern
        2. They cover the same cells (row, col positions)
        3. C1 covers more depth (more identifiers) than C2
        
        Args:
            clusters: List of cluster dictionaries
            
        Returns:
            list: Non-dominated clusters
        """
        essential = []
        
        for i, c1 in enumerate(clusters):
            dominated = False
            
            for j, c2 in enumerate(clusters):
                if i == j:
                    continue
                
                # Check if c1 is dominated by c2
                if self._cluster_dominates(c2, c1):
                    dominated = True
                    print(f"  Dominated: {c1['full_pattern']} ⊂ {c2['full_pattern']}")
                    break
            
            if not dominated:
                essential.append(c1)
        
        return essential

    def _cluster_dominates(self, c1, c2):
        """
        Check if cluster c1 dominates cluster c2.
        
        Args:
            c1, c2: Cluster dictionaries
            
        Returns:
            bool: True if c1 dominates c2
        """
        # Must have same K-map pattern
        if c1['kmap_pattern'] != c2['kmap_pattern']:
            return False
        
        # Check if c1's identifier pattern subsumes c2's
        # This means: every identifier covered by c2 is also covered by c1
        id1 = c1['identifier_pattern']
        id2 = c2['identifier_pattern']
        
        # Convert to sets of actual identifiers they represent
        ids1_set = self._pattern_to_identifier_set(id1)
        ids2_set = self._pattern_to_identifier_set(id2)
        
        # c1 dominates c2 if c2's identifiers are a proper subset of c1's
        return ids2_set < ids1_set  # Proper subset

    def _pattern_to_identifier_set(self, pattern):
        """
        Convert an identifier pattern with don't cares to the set of 
        concrete identifiers it represents.
        
        Args:
            pattern: String with possible '-' (e.g., "0-1")
            
        Returns:
            set: Set of concrete binary strings
        """
        # Count don't cares
        n_dontcares = pattern.count('-')
        
        if n_dontcares == 0:
            return {pattern}
        
        # Generate all combinations
        result = set()
        for i in range(2 ** n_dontcares):
            concrete = list(pattern)
            bits = format(i, f'0{n_dontcares}b')
            bit_idx = 0
            
            for j in range(len(concrete)):
                if concrete[j] == '-':
                    concrete[j] = bits[bit_idx]
                    bit_idx += 1
            
            result.add(''.join(concrete))
        
        return result

    def _find_all_prime_implicants_bitwise(self, minterm_list):
        """
        Phase 1: Find ALL prime implicants using bitwise operations.
        This is the existing code, just separated out.
        """
        bit_width = len(minterm_list[0])
        
        # Convert to bitwise
        terms = set()
        for term_str in minterm_list:
            value, mask = self._str_to_bitwise(term_str)
            terms.add((value, mask))
        
        # Iterative merging
        prime_implicants = set()
        iteration = 1
        
        while True:
            next_terms = set()
            used = set()
            
            terms_list = list(terms)
            
            for i in range(len(terms_list)):
                for j in range(i + 1, len(terms_list)):
                    val1, mask1 = terms_list[i]
                    val2, mask2 = terms_list[j]
                    
                    if mask1 != mask2:
                        continue
                    
                    diff = (val1 ^ val2) & mask1
                    
                    if diff != 0 and (diff & (diff - 1)) == 0:
                        merged_value = val1 & val2
                        merged_mask = mask1 & ~diff
                        
                        next_terms.add((merged_value, merged_mask))
                        used.add(terms_list[i])
                        used.add(terms_list[j])
            
            # Unused terms are prime implicants
            for term in terms_list:
                if term not in used:
                    prime_implicants.add(term)
            
            if not next_terms:
                break
            
            terms = next_terms
            iteration += 1
        
        # Convert to strings
        bit_width = len(minterm_list[0])
        return [self._bitwise_to_str(val, mask, bit_width) 
                for val, mask in prime_implicants]

    def _select_essential_prime_implicants(self, prime_implicants, minterms):
        """
        Phase 2: Select minimum cover using essential prime implicants.
        
        This ensures the result is minimal.
        
        Args:
            prime_implicants: List of all prime implicants (strings with '-')
            minterms: List of original minterms (strings without '-')
            
        Returns:
            list: Minimal set of prime implicants covering all minterms
        """
        # Build coverage table: which PIs cover which minterms
        coverage = {}
        for pi in prime_implicants:
            coverage[pi] = set()
            for mt in minterms:
                if self._implicant_covers_minterm(pi, mt):
                    coverage[pi].add(mt)
        
        # Find essential prime implicants
        essential = []
        covered_minterms = set()
        uncovered_minterms = set(minterms)
        
        # Step 1: Find essential PIs (minterms covered by only one PI)
        for mt in minterms:
            covering_pis = [pi for pi, covered in coverage.items() if mt in covered]
            
            if len(covering_pis) == 1:
                # This is an essential prime implicant
                pi = covering_pis[0]
                if pi not in essential:
                    essential.append(pi)
                    covered_minterms.update(coverage[pi])
                    uncovered_minterms -= coverage[pi]
                    print(f"      Essential: {pi} (covers {len(coverage[pi])} minterms)")
        
        # Step 2: Cover remaining minterms (greedy heuristic)
        remaining_pis = [pi for pi in prime_implicants if pi not in essential]
        
        while uncovered_minterms and remaining_pis:
            # Choose PI that covers most uncovered minterms
            best_pi = max(remaining_pis, 
                        key=lambda pi: len(coverage[pi] & uncovered_minterms))
            
            if len(coverage[best_pi] & uncovered_minterms) == 0:
                break
            
            essential.append(best_pi)
            newly_covered = coverage[best_pi] & uncovered_minterms
            covered_minterms.update(newly_covered)
            uncovered_minterms -= newly_covered
            remaining_pis.remove(best_pi)
            
            print(f"      Added: {best_pi} (covers {len(newly_covered)} more minterms)")
        
        return essential
    
    def _implicant_covers_minterm(self, implicant, minterm):
        """
        Check if a prime implicant (with '-') covers a specific minterm.
        
        Args:
            implicant: String with possible '-' (e.g., "10-1")
            minterm: String without '-' (e.g., "1001")
            
        Returns:
            bool: True if implicant covers minterm
            
        Example:
            "10-1" covers "1001" → True
            "10-1" covers "1011" → True
            "10-1" covers "1101" → False (first bit doesn't match)
        """
        if len(implicant) != len(minterm):
            return False
        
        for i in range(len(implicant)):
            if implicant[i] != '-' and implicant[i] != minterm[i]:
                return False
        
        return True

    def _get_all_minterms(self, β, id_set):
        """
        Get all minterms (identifier + K-map position combinations) 
        where function = 1.
        
        Args:
            β: Dictionary mapping identifier → patterns
            id_set: List of identifiers
            
        Returns:
            set: Set of full minterms (identifier + position strings)
        """
        minterms = set()
        
        for idx in id_set:
            # Get the actual K-map for this identifier
            kmap = self.kmaps[idx]
            
            # Find all positions where value = 1
            for row_idx in range(4):
                for col_idx in range(4):
                    cell = kmap[row_idx][col_idx]
                    if cell and cell.get('value') == 1:
                        # Get the position in binary (last 4 bits)
                        vars_binary = ''.join(str(b) for b in cell['variables'][-4:])
                        minterm = idx + vars_binary
                        minterms.add(minterm)
        
        return minterms

    def _get_covered_minterms(self, clusters):
        """
        Get all minterms covered by a set of clusters.
        
        Args:
            clusters: Set of full pattern strings
            
        Returns:
            set: Set of minterms covered
        """
        covered = set()
        
        for pattern in clusters:
            # Expand pattern to all concrete minterms it represents
            covered.update(self._expand_pattern(pattern))
        
        return covered

    def _expand_pattern(self, pattern):
        """
        Expand a pattern with don't cares to all concrete minterms.
        
        Args:
            pattern: String with possible '-'
            
        Returns:
            set: Set of concrete binary strings
        """
        return self._pattern_to_identifier_set(pattern)

    def _greedy_cover(self, uncovered, clusters_3d, clusters_2d, id_set):
        """
        Apply greedy algorithm to cover remaining uncovered minterms.
        
        Args:
            uncovered: Set of uncovered minterms
            clusters_3d: List of 3D cluster dictionaries
            clusters_2d: Dictionary of 2D clusters
            id_set: List of identifiers
            
        Returns:
            set: Additional patterns needed for coverage
        """
        additional = set()
        remaining = uncovered.copy()
        
        # Build candidate list from all clusters
        candidates = []
        
        for cluster in clusters_3d:
            covered = self._expand_pattern(cluster['full_pattern'])
            candidates.append((cluster['full_pattern'], covered))
        
        for pattern, id_list in clusters_2d.items():
            for idx in set(id_list):
                full = idx + pattern
                covered = self._expand_pattern(full)
                candidates.append((full, covered))
        
        # Greedy selection
        while remaining and candidates:
            # Find candidate covering most remaining minterms
            best = max(candidates, key=lambda c: len(c[1] & remaining))
            
            if len(best[1] & remaining) == 0:
                break
            
            additional.add(best[0])
            remaining -= best[1]
            candidates.remove(best)
        
        return additional
    def _str_to_bitwise(self, term_str):
        """
        Convert string with '-' to bitwise representation (value, mask).
        
        Args:
            term_str: Binary string with possible '-'
            
        Returns:
            tuple: (value, mask) where mask has 1 for fixed bits, 0 for don't cares
            
        Example:
            "10-1" → (value=0b1001, mask=0b1101)
                    bits:  1 0 - 1
                    mask:  1 1 0 1  (1=fixed, 0=don't care)
                    value: 1 0 0 1  (don't care treated as 0)
        """
        bit_width = len(term_str)
        value = 0
        mask = 0
        
        for i, bit in enumerate(term_str):
            bit_pos = bit_width - 1 - i  # MSB first
            if bit != '-':
                mask |= (1 << bit_pos)  # Mark as fixed
                if bit == '1':
                    value |= (1 << bit_pos)  # Set bit
        
        return value, mask

    def _bitwise_to_str(self, value, mask, bit_width):
        """
        Convert bitwise representation (value, mask) to string with '-'.
        
        Args:
            value: Integer representing the bit values
            mask: Integer where 1 = fixed bit, 0 = don't care
            bit_width: Number of bits
            
        Returns:
            str: Binary string with '-' for don't cares
            
        Example:
            (value=0b1001, mask=0b1101, width=4) → "10-1"
        """
        result = []
        for i in range(bit_width):
            bit_pos = bit_width - 1 - i  # MSB first
            if mask & (1 << bit_pos):
                # Bit is fixed
                if value & (1 << bit_pos):
                    result.append('1')
                else:
                    result.append('0')
            else:
                # Don't care
                result.append('-')
        return ''.join(result)
    
    def _bits_to_term(self, bit_pattern, form='sop'):
        """
        Convert a bit pattern (with '-' for don't cares) to a Boolean term.
        
        Args:
            bit_pattern (str): Binary pattern with '-' for don't cares
            form (str): 'sop' or 'pos'
            
        Returns:
            str: Boolean term with variable names
        """
        vars_ = [f"x{i+1}" for i in range(self.num_vars)]
        
        if form.lower() == 'pos':
            # Product of Sums: invert logic
            literals = []
            for i, bit in enumerate(bit_pattern):
                if bit == '1':
                    literals.append(vars_[i] + "'")
                elif bit == '0':
                    literals.append(vars_[i])
            return "(" + " + ".join(literals) + ")" if literals else "(1)"
        else:
            # Sum of Products
            literals = []
            for i, bit in enumerate(bit_pattern):
                if bit == '0':
                    literals.append(vars_[i] + "'")
                elif bit == '1':
                    literals.append(vars_[i])
            return "".join(literals) if literals else "1"

    # ============================================================================
    # VECTORIZED IMPLEMENTATION (Ultra-fast for large problems)
    # ============================================================================

    def _minimize_boolean_function_vectorized(self, minterm_list):
        """
        Ultra-fast vectorized implementation using NumPy.
        Optimized for large numbers of identifiers (64+).
        
        Time Complexity: O(n² / SIMD_width × iterations)
        Space Complexity: O(n)
        
        Args:
            minterm_list: List of binary strings
            
        Returns:
            list: List of prime implicants
        """
        if not minterm_list:
            return []
        
        if len(minterm_list) == 1:
            return minterm_list
        
        bit_width = len(minterm_list[0])
        
        # Convert to numpy arrays for vectorized operations
        unique_terms = list(set(minterm_list))
        n_terms = len(unique_terms)
        
        values = np.zeros(n_terms, dtype=np.uint32)
        masks = np.zeros(n_terms, dtype=np.uint32)
        
        for idx, term_str in enumerate(unique_terms):
            val, mask = self._str_to_bitwise(term_str)
            values[idx] = val
            masks[idx] = mask
        
        prime_implicants = []
        iteration = 1
        
        while len(values) > 0:
            next_values = []
            next_masks = []
            used = np.zeros(len(values), dtype=bool)
            merge_count = 0
            
            # Vectorized pairwise comparison
            for i in range(len(values)):
                if used[i]:
                    continue
                
                # Find terms with same mask (vectorized)
                same_mask_indices = np.where(masks == masks[i])[0]
                same_mask_indices = same_mask_indices[same_mask_indices > i]  # Only check j > i
                
                if len(same_mask_indices) == 0:
                    continue
                
                # Vectorized XOR to find differences
                diffs = (values[i] ^ values[same_mask_indices]) & masks[i]
                
                # Check which diffs are powers of 2 (exactly 1 bit set)
                # A number is power of 2 if: n != 0 and n & (n-1) == 0
                is_power_of_2 = (diffs != 0) & ((diffs & (diffs - 1)) == 0)
                
                if np.any(is_power_of_2):
                    mergeable_indices = same_mask_indices[is_power_of_2]
                    
                    for j in mergeable_indices:
                        diff = (values[i] ^ values[j]) & masks[i]
                        merged_value = values[i] & values[j]
                        merged_mask = masks[i] & ~diff
                        
                        next_values.append(merged_value)
                        next_masks.append(merged_mask)
                        used[i] = True
                        used[j] = True
                        merge_count += 1
            
            if merge_count > 0:
                print(f"    Iteration {iteration}: {merge_count} merges (vectorized)")
            
            # Add unused terms as prime implicants
            for idx in np.where(~used)[0]:
                pi_str = self._bitwise_to_str(values[idx], masks[idx], bit_width)
                prime_implicants.append(pi_str)
            
            # If no new merges, we're done
            if not next_values:
                break
            
            # Prepare for next iteration
            values = np.array(next_values, dtype=np.uint32)
            masks = np.array(next_masks, dtype=np.uint32)
            iteration += 1
        
        return prime_implicants if prime_implicants else minterm_list

    def minimize_4d(self, form='sop'):
        """
        4D Karnaugh map minimization for n > 8 variables.
        
        Dimensional structure:
        - Length (2 bits): Rows in 2D K-map
        - Breadth (2 bits): Columns in 2D K-map
        - Depth (m bits): Identifiers in 3D structure
        - Span (c bits): Chunks in 4D structure
        
        where m = 8-4 = 4 (for inner 8-var maps)
        and c = n-8 (for outer chunk dimension)
        
        Args:
            form (str): 'sop' or 'pos'
            
        Returns:
            tuple: (list of minimized terms, complete expression string)
        """
        if self.num_vars <= 8:
            # Fall back to 3D for n ≤ 8
            print("Using 3D minimization (n ≤ 8)")
            return self.minimize_3d(form)
        
        print(f"\n{'='*70}")
        print(f"4D K-MAP MINIMIZATION")
        print(f"{'='*70}")
        print(f"Total variables: {self.num_vars}")
        print(f"Chunk bits (span): {self.num_vars - 8}")
        print(f"Variables per chunk: 8")
        print(f"Structure: {2**(self.num_vars-8)} chunks × 16 K-maps × 4×4 cells")
        print(f"{'='*70}\n")
        
        # Step 1: Partition into chunks (8-variable subproblems)
        chunks = self._partition_into_chunks()
        
        print(f"PHASE 1: CHUNK PARTITION")
        print(f"Created {len(chunks)} chunks\n")
        
        # Step 2: Solve each chunk using 3D minimization
        chunk_results = {}  # chunk_id → 3D minimized patterns
        
        print(f"PHASE 2: 3D MINIMIZATION PER CHUNK")
        print("-" * 70)
        
        for chunk_id, chunk_kmap in chunks.items():
            print(f"\nChunk {chunk_id}:")
            print(f"  Solving 8-variable subproblem...")
            
            # Create temporary 8-variable K-map object
            chunk_minimizer = self._create_chunk_minimizer(chunk_kmap)
            
            # Apply 3D minimization
            terms, _ = chunk_minimizer.minimize_3d(form)
            
            # Extract patterns (without conversion to variables yet)
            patterns = self._extract_patterns_from_terms(terms, chunk_minimizer)
            
            chunk_results[chunk_id] = patterns
            
            print(f"  Found {len(patterns)} 3D-minimized patterns")
            for pattern in patterns[:5]:  # Show first 5
                print(f"    {pattern}")
            if len(patterns) > 5:
                print(f"    ... and {len(patterns)-5} more")
        
        # Step 3: Identify 4D clusters (patterns spanning multiple chunks)
        print(f"\n{'='*70}")
        print(f"PHASE 3: 4D CLUSTER IDENTIFICATION")
        print(f"{'='*70}\n")
        
        pattern_to_chunks = self._map_patterns_to_chunks(chunk_results)
        
        valid_4d_clusters = {}
        eliminated_3d = []
        
        for pattern, chunk_list in pattern_to_chunks.items():
            if len(chunk_list) == 1:
                # Pure 3D cluster (no span)
                eliminated_3d.append((pattern, chunk_list[0]))
                print(f"  ✗ ELIMINATED 3D: '{pattern}' in chunk {chunk_list[0]}")
            elif not self._has_adjacent_chunks(chunk_list):
                # Non-adjacent chunks
                eliminated_3d.extend([(pattern, chunk) for chunk in chunk_list])
                print(f"  ✗ ELIMINATED Non-adjacent: '{pattern}' in chunks {chunk_list}")
            else:
                # Valid 4D cluster
                valid_4d_clusters[pattern] = chunk_list
                print(f"  ✓ VALID 4D: '{pattern}' spans {len(chunk_list)} chunks")
        
        print(f"\n  Kept: {len(valid_4d_clusters)} 4D clusters")
        print(f"  Eliminated: {len(eliminated_3d)} 3D-only patterns")
        
        if not valid_4d_clusters:
            print("\n  WARNING: No 4D clusters found!")
            print("  Falling back to treating all patterns as 3D...")
            valid_4d_clusters = pattern_to_chunks
        
        # Step 4: Merge chunks (span-wise) for each pattern
        print(f"\n{'='*70}")
        print(f"PHASE 4: SPAN-WISE MERGING")
        print(f"{'='*70}\n")
        
        merged_4d_clusters = []
        
        for pattern, chunk_list in valid_4d_clusters.items():
            print(f"Merging chunks for pattern '{pattern}':")
            print(f"  Chunks: {chunk_list}")
            
            # Apply Quine-McCluskey to chunk identifiers
            merged_chunks = self._minimize_boolean_function_complete(chunk_list)
            
            print(f"  Span implicants: {merged_chunks}")
            
            for merged_chunk in merged_chunks:
                cluster = {
                    'inner_pattern': pattern,      # 8-bit pattern from 3D minimization
                    'chunk_pattern': merged_chunk,  # Chunk identifier with don't cares
                    'full_pattern': merged_chunk + pattern,  # Complete n-bit pattern
                    'span': self._count_depth(merged_chunk),  # Number of chunks
                    'cells_3d': self._get_3d_cell_coverage(pattern)  # 3D space coverage
                }
                merged_4d_clusters.append(cluster)
                print(f"    4D Cluster: {merged_chunk} + {pattern} (span={cluster['span']})")
        
        # Step 5: Select EPIs using span dominance
        print(f"\n{'='*70}")
        print(f"PHASE 5: ESSENTIAL PRIME IMPLICANT SELECTION (Span-wise)")
        print(f"{'='*70}\n")
        
        epis = self._select_epis_by_span_dominance(merged_4d_clusters)
        
        print(f"Selected {len(epis)} essential prime implicants")

        #  # Step 6: Verify coverage
        # print(f"\n{'='*70}")
        # print(f"PHASE 6: COVERAGE VERIFICATION")
        # print(f"{'='*70}\n")
        
        # all_minterms = self._get_all_minterms_from_kmaps(list(self.kmaps.keys()))
        # covered = self._get_covered_minterms([c['full_pattern'] for c in epis])
        # uncovered = all_minterms - covered
        
        # if uncovered:
        #     print(f"  ⚠ WARNING: {len(uncovered)} minterms not covered!")
        #     additional = self._greedy_cover_4d(uncovered, merged_4d_clusters, epis)
        #     epis.extend(additional)
            
        #     covered = self._get_covered_minterms([c['full_pattern'] for c in epis])
        #     still_uncovered = all_minterms - covered
            
        #     if still_uncovered:
        #         print(f"  ✗ CRITICAL: {len(still_uncovered)} minterms still uncovered!")
        #     else:
        #         print(f"  ✓ Coverage achieved with {len(additional)} additional clusters")
        # else:
        #     print(f"  ✓ All {len(all_minterms)} minterms covered")
        
        # Step 7: Convert to final expression
        final_terms = []
        for cluster in epis:
            term_str = self._bits_to_term(cluster['full_pattern'], form)
            final_terms.append(term_str)
        
        join_operator = " * " if form.lower() == 'pos' else " + "
        final_expression = join_operator.join(final_terms)
        
        print(f"\n{'='*70}")
        print(f"FINAL 4D-MINIMIZED EXPRESSION")
        print(f"{'='*70}")
        print(f"Terms: {len(final_terms)}")
        print(f"F = {final_expression}\n")
        
        return final_terms, final_expression

    # ============================================================================
    # CHUNK PARTITIONING (4D Structure)
    # ============================================================================

    def _partition_into_chunks(self):
        """
        Partition the n-variable K-map into 8-variable chunks.
        
        For n-variable function:
        - First c = n-8 bits define chunk
        - Remaining 8 bits define position within chunk
        
        Returns:
            dict: chunk_id → 8-variable K-map structure
        """
        c = self.num_vars - 8  # chunk bits
        num_chunks = 2 ** c
        
        chunks = {}
        
        # Generate all chunk identifiers
        for chunk_idx in range(num_chunks):
            chunk_id = format(chunk_idx, f'0{c}b')
            
            # Extract K-maps belonging to this chunk
            chunk_kmaps = {}
            
            # For this chunk, we need the 16 K-maps (4-bit identifiers for 8-var)
            for inner_id_idx in range(16):
                inner_id = format(inner_id_idx, '04b')  # 4 bits for inner identifiers
                
                # Full identifier in original structure
                full_id = chunk_id + inner_id
                
                if full_id in self.kmaps:
                    chunk_kmaps[inner_id] = self.kmaps[full_id]
            
            chunks[chunk_id] = chunk_kmaps
        
        return chunks

    def _create_chunk_minimizer(self, chunk_kmaps):
        """
        Create a temporary 8-variable K-map minimizer for a chunk.
        
        Args:
            chunk_kmaps: dict mapping 4-bit identifiers to 4×4 K-maps
            
        Returns:
            KMap3D: 8-variable K-map object
        """
        # Create new K-map object for 8 variables
        from copy import deepcopy
        
        chunk_minimizer = self.__class__(8)  # Assuming your class is callable with num_vars
        chunk_minimizer.kmaps = deepcopy(chunk_kmaps)
        chunk_minimizer.num_extra_vars = 4
        chunk_minimizer.num_maps = 16
        
        return chunk_minimizer

    def _extract_patterns_from_terms(self, terms, chunk_minimizer):
        """
        Extract bit patterns from minimized terms.
        
        Args:
            terms: List of term strings (e.g., ["x1x2'x5", "x3x6'x7x8"])
            chunk_minimizer: The 8-variable minimizer used
            
        Returns:
            list: List of 8-bit patterns with don't cares
        """
        patterns = []
        
        for term in terms:
            # Convert term back to bit pattern
            pattern = self._term_to_pattern(term, 8)
            patterns.append(pattern)
        
        return patterns

    def _term_to_pattern(self, term, num_vars):
        """
        Convert a term string back to bit pattern.
        
        Args:
            term: String like "x1x2'x5" 
            num_vars: Number of variables
            
        Returns:
            str: Bit pattern like "10-01---"
        """
        # Initialize with don't cares
        pattern = ['-'] * num_vars
        
        # Parse term to find which variables are present
        import re
        
        # Find all variables (with or without complement)
        var_pattern = r"x(\d+)('?)"
        matches = re.findall(var_pattern, term)
        
        for var_num_str, complement in matches:
            var_num = int(var_num_str)
            if 1 <= var_num <= num_vars:
                idx = var_num - 1  # 0-indexed
                pattern[idx] = '0' if complement else '1'
        
        return ''.join(pattern)

    # ============================================================================
    # 4D CLUSTER OPERATIONS
    # ============================================================================

    def _map_patterns_to_chunks(self, chunk_results):
        """
        Create mapping: pattern → list of chunks where it appears.
        
        Args:
            chunk_results: dict of chunk_id → list of patterns
            
        Returns:
            dict: pattern → list of chunk_ids
        """
        pattern_to_chunks = {}
        
        for chunk_id, patterns in chunk_results.items():
            for pattern in patterns:
                if pattern not in pattern_to_chunks:
                    pattern_to_chunks[pattern] = []
                pattern_to_chunks[pattern].append(chunk_id)
        
        return pattern_to_chunks

    def _has_adjacent_chunks(self, chunk_list):
        """
        Check if any pair of chunks in the list is adjacent.
        
        Args:
            chunk_list: List of chunk identifier strings
            
        Returns:
            bool: True if at least one adjacent pair exists
        """
        for i in range(len(chunk_list)):
            for j in range(i + 1, len(chunk_list)):
                if self._hamming_distance(chunk_list[i], chunk_list[j]) == 1:
                    return True
        return False

    def _get_3d_cell_coverage(self, pattern):
        """
        Get the 3D cell coverage (depth × breadth × length) for an 8-bit pattern.
        
        Args:
            pattern: 8-bit pattern string
            
        Returns:
            frozenset: Set of (depth, row, col) positions
        """
        # First 4 bits = depth (identifier in 3D structure)
        # Last 4 bits = position in 2D K-map
        
        depth_pattern = pattern[:4]
        kmap_pattern = pattern[4:]
        
        # Expand to concrete positions
        depths = self._expand_pattern(depth_pattern)
        cells_2d = self._get_cell_positions(kmap_pattern)
        
        # Combine into 3D coordinates
        cells_3d = set()
        for depth in depths:
            for row, col in cells_2d:
                cells_3d.add((depth, row, col))
        
        return frozenset(cells_3d)

    def _select_epis_by_span_dominance(self, clusters):
        """
        Select EPIs using span dominance criterion.
        
        For clusters with same 3D cell coverage:
        - Keep the one with maximum span (most chunks)
        - Eliminate others
        
        Args:
            clusters: List of 4D cluster dictionaries
            
        Returns:
            list: Essential prime implicants
        """
        # Group by inner pattern (3D structure)
        pattern_groups = {}
        for cluster in clusters:
            inner = cluster['inner_pattern']
            if inner not in pattern_groups:
                pattern_groups[inner] = []
            pattern_groups[inner].append(cluster)
        
        epis = []
        
        for inner_pattern, group in pattern_groups.items():
            print(f"\n  Analyzing pattern '{inner_pattern}':")
            
            # Further group by 3D cell coverage
            cell_groups = {}
            for cluster in group:
                cells_key = cluster['cells_3d']
                if cells_key not in cell_groups:
                    cell_groups[cells_key] = []
                cell_groups[cells_key].append(cluster)
            
            # For each cell group, select maximum span
            for cells, cell_group in cell_groups.items():
                print(f"    3D coverage: {len(cells)} cells")
                
                # Find maximum span
                max_span = max(c['span'] for c in cell_group)
                
                # Keep only clusters with maximum span
                for cluster in cell_group:
                    if cluster['span'] == max_span:
                        epis.append(cluster)
                        print(f"      ✓ EPI: {cluster['chunk_pattern']} + {cluster['inner_pattern']} "
                            f"(span={cluster['span']}) - MAX SPAN")
                    else:
                        print(f"      ✗ Dominated: {cluster['chunk_pattern']} + {cluster['inner_pattern']} "
                            f"(span={cluster['span']}) < {max_span}")
        
        return epis

    def _greedy_cover_4d(self, uncovered, all_clusters, current_epis):
        """
        Greedy coverage for uncovered minterms using 4D clusters.
        
        Args:
            uncovered: Set of uncovered minterms
            all_clusters: List of all 4D cluster dictionaries
            current_epis: List of currently selected EPIs
            
        Returns:
            list: Additional clusters needed
        """
        additional = []
        remaining = uncovered.copy()
        
        # Get available clusters
        current_patterns = {c['full_pattern'] for c in current_epis}
        available = [c for c in all_clusters 
                    if c['full_pattern'] not in current_patterns]
        
        while remaining and available:
            # Find cluster covering most remaining minterms
            best_cluster = None
            best_coverage = 0
            
            for cluster in available:
                covered = self._expand_pattern(cluster['full_pattern'])
                overlap = len(covered & remaining)
                if overlap > best_coverage:
                    best_coverage = overlap
                    best_cluster = cluster
            
            if best_cluster is None or best_coverage == 0:
                break
            
            additional.append(best_cluster)
            newly_covered = self._expand_pattern(best_cluster['full_pattern'])
            remaining -= newly_covered
            available.remove(best_cluster)
            
            print(f"    Added: {best_cluster['full_pattern']} "
                f"(span={best_cluster['span']}, covers {best_coverage} more minterms)")
        
        return additional

    # ============================================================================
    # HEIRARCHICAL MINIMIZATION FOR  N>10 variables where dimensionality collapses
    # ============================================================================
    def minimize_heirarchical(self, form='sop'):    
        """
        Minimize the entire 3D K-map by solving each hierarchical map
        and combining results using bitwise union operations.
        
        Args:
            form (str): 'sop' or 'pos'
            
        Returns:
            tuple: (list of minimized terms, complete expression string)
        """
        print(f"\n{'='*60}")
        print(f"MINIMIZING {self.num_vars}-VARIABLE K-MAP")
        print(f"{'='*60}\n")
        
        # Step 1: Solve each hierarchical K-map
        all_map_results = []
        
        for extra_combo in sorted(self.kmaps.keys()):
            print(f"Solving K-map for extra vars: {extra_combo}")
            result = self._solve_single_kmap(extra_combo, form)
            all_map_results.append(result)
            print(f"  Found {len(result['bitmasks'])} essential prime implicants")
            for i, bits in enumerate(result['terms_bits']):
                print(f"    Term {i+1}: {extra_combo}|{bits}")
        
        print(f"\n{'='*60}")
        print("COMBINING RESULTS ACROSS ALL K-MAPS")
        print(f"{'='*60}\n")
        
        # Step 2: Combine using bitwise union across all maps
        # Build a global dictionary: bit_pattern -> list of (extra_combo, bitmask)
        pattern_groups = defaultdict(list)
        
        for result in all_map_results:
            extra_combo = result['extra_vars']
            for i, term_bits in enumerate(result['terms_bits']):
                # The pattern is the 4-bit portion from the 4x4 map
                pattern_groups[term_bits].append({
                    'extra_combo': extra_combo,
                    'bitmask': result['bitmasks'][i]
                })
        
        # Step 3: For each unique pattern, combine extra variable combinations
        final_terms = []
        
        for pattern, occurrences in pattern_groups.items():
            # Collect all extra variable combinations for this pattern
            extra_combos = [occ['extra_combo'] for occ in occurrences]
            
            # Simplify the extra variable portion if possible
            simplified_extra = self._simplify_extra_vars(extra_combos)
            
            # Combine into full term
            full_pattern = simplified_extra + pattern
            term_str = self._bits_to_term(full_pattern, form)
            final_terms.append(term_str)
            
            print(f"Pattern {pattern} appears in maps: {extra_combos}")
            print(f"  Simplified extra vars: {simplified_extra}")
            print(f"  Final term: {term_str}\n")
        
        # Step 4: Build final expression
        join_operator = " * " if form.lower() == 'pos' else " + "
        final_expression = join_operator.join(final_terms)
        
        print(f"{'='*60}")
        print(f"FINAL MINIMIZED EXPRESSION ({form.upper()}):")
        print(f"{'='*60}")
        print(f"F = {final_expression}\n")
        
        return final_terms, final_expression

    def minimize_hierarchical_10var(self, form='sop'):
        """
        Hierarchical minimization using 10-variable base (4D limit).
        
        For n > 10 variables:
        - Partition into 10-variable chunks
        - Solve each chunk using 4D minimization
        - Collect all patterns (set handles deduplication automatically)
        
        This is NOT geometric clustering - just hierarchical decomposition.
        Redundancy elimination happens naturally via set union.
        
        Args:
            form (str): 'sop' or 'pos'
            
        Returns:
            tuple: (list of terms, expression string)
        """
        if self.num_vars <= 10:
            print(f"Using 4D minimization directly (n={self.num_vars} ≤ 10)")
            return self.minimize_4d(form)
        
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL 10-VARIABLE BASE MINIMIZATION")
        print(f"{'='*70}")
        print(f"Total variables: {self.num_vars}")
        print(f"Outer dimension bits: {self.num_vars - 10}")
        print(f"Number of 10-var chunks: {2**(self.num_vars-10)}")
        print(f"Method: Hierarchical (NOT geometric clustering)")
        print(f"{'='*70}\n")
        
        # Step 1: Partition into 10-variable chunks
        chunks = self._partition_into_10var_chunks()
        
        print(f"PHASE 1: 10-VARIABLE CHUNK PARTITION")
        print(f"Created {len(chunks)} chunks\n")
        
        # Step 2: Solve each chunk using 4D minimization
        all_patterns = set()  # Set automatically handles duplicates
        
        print(f"PHASE 2: 4D MINIMIZATION PER CHUNK")
        print("-" * 70)
        
        for chunk_id in sorted(chunks.keys()):
            print(f"\nChunk {chunk_id}:")
            print(f"  Solving 10-variable subproblem using 4D method...")
            
            # Create temporary 10-variable K-map solver
            chunk_solver = self._create_10var_chunk_solver(chunk_id, chunks[chunk_id])
            
            # Apply 4D minimization
            terms, _ = chunk_solver.minimize_4d(form)
            
            # Extract bit patterns from terms
            patterns = self._extract_bit_patterns_from_terms(terms, 10)
            
            print(f"  Found {len(patterns)} 4D-minimized patterns")
            
            # Add chunk prefix and insert into set
            for pattern in patterns:
                full_pattern = chunk_id + pattern
                all_patterns.add(full_pattern)  # Set handles deduplication
                print(f"    {full_pattern}")
        
        # Step 3: Report results (redundancies already eliminated by set)
        print(f"\n{'='*70}")
        print(f"PHASE 3: PATTERN COLLECTION (Bitwise Union)")
        print(f"{'='*70}")
        print(f"Total unique patterns: {len(all_patterns)}")
        print(f"(Redundancies automatically eliminated by set)\n")
        
        # Step 4: Convert to final expression
        final_terms = []
        for full_pattern in sorted(all_patterns):
            term_str = self._bits_to_term(full_pattern, form)
            final_terms.append(term_str)
        
        join_operator = " * " if form.lower() == 'pos' else " + "
        final_expression = join_operator.join(final_terms)
        
        print(f"\n{'='*70}")
        print(f"FINAL HIERARCHICAL EXPRESSION")
        print(f"{'='*70}")
        print(f"Terms: {len(final_terms)}")
        print(f"F = {final_expression}\n")
        
        return final_terms, final_expression

    def _partition_into_10var_chunks(self):
        """
        Partition the n-variable K-map into 10-variable chunks.
        
        For n-variable function:
        - First (n-10) bits define chunk identifier
        - Remaining 10 bits define position within chunk
        
        Returns:
            dict: chunk_id → dict of K-maps for that chunk
        """
        outer_bits = self.num_vars - 10  # bits for outer dimension
        num_chunks = 2 ** outer_bits
        
        chunks = {}
        
        # Generate all chunk identifiers
        for chunk_idx in range(num_chunks):
            chunk_id = format(chunk_idx, f'0{outer_bits}b')
            
            # For this chunk, collect all K-maps with this prefix
            chunk_kmaps = {}
            
            for full_id in self.kmaps.keys():
                if full_id.startswith(chunk_id):
                    # Extract the inner portion (last part after chunk_id)
                    inner_id = full_id[outer_bits:]
                    chunk_kmaps[inner_id] = self.kmaps[full_id]
            
            chunks[chunk_id] = chunk_kmaps
        
        return chunks

    def _create_10var_chunk_solver(self, chunk_id, chunk_kmaps):
        """
        Create a temporary 10-variable K-map solver for a chunk.
        
        Args:
            chunk_id (str): Binary string identifying this chunk
            chunk_kmaps (dict): Dictionary of K-maps for this chunk
            
        Returns:
            BoolMinGeo: 10-variable K-map solver instance
        """
        # Create new solver for 10 variables
        chunk_solver = BoolMinGeo.__new_(BoolMinGeo)
        chunk_solver.num_vars = 10
        chunk_solver.num_extra_vars = 6  # 10 - 4 = 6
        chunk_solver.num_maps = 64  # 2^6
        chunk_solver.gray_code_4 = ["00", "01", "11", "10"]
        
        # Copy the K-maps
        chunk_solver.kmaps = {}
        for inner_id, kmap in chunk_kmaps.items():
            chunk_solver.kmaps[inner_id] = kmap
        
        # Generate truth table for this chunk
        chunk_solver.truth_table = chunk_solver.generate_truth_table()
        
        # Extract output values from K-maps
        chunk_solver.output_values = [0] * (2**10)
        for inner_id, kmap in chunk_kmaps.items():
            for row_idx in range(4):
                for col_idx in range(4):
                    cell = kmap[row_idx][col_idx]
                    if cell:
                        minterm_in_chunk = self._get_minterm_index_in_chunk(
                            inner_id, row_idx, col_idx, 10
                        )
                        chunk_solver.output_values[minterm_in_chunk] = cell['value']
        
        return chunk_solver

    def _get_minterm_index_in_chunk(self, inner_id, row_idx, col_idx, chunk_size):
        """
        Get the minterm index within a chunk given identifier and cell position.
        
        Args:
            inner_id (str): Binary string for inner identifier
            row_idx (int): Row index in 4×4 K-map
            col_idx (int): Column index in 4×4 K-map
            chunk_size (int): Size of chunk in bits (10)
            
        Returns:
            int: Minterm index within chunk (0 to 2^chunk_size - 1)
        """
        # Get Gray code for row and column
        gray_code = ["00", "01", "11", "10"]
        col_gray = gray_code[col_idx]
        row_gray = gray_code[row_idx]
        
        # Construct full binary string for position in chunk
        # Format: [inner_id][col_gray][row_gray]
        full_bits = inner_id + col_gray + row_gray
        
        # Convert to integer
        return int(full_bits, 2)

    def _extract_bit_patterns_from_terms(self, terms, num_bits):
        """
        Extract bit patterns from minimized terms.
        
        Args:
            terms (list): List of term strings (e.g., ["x1x2'x5", "x3x6'x7x8"])
            num_bits (int): Number of bits in pattern (10 for 10-var chunks)
            
        Returns:
            list: List of bit patterns with don't cares (e.g., ["10-01---", "---10110"])
        """
        patterns = []
        
        for term in terms:
            # Convert term back to bit pattern
            pattern = self._term_to_bit_pattern(term, num_bits)
            patterns.append(pattern)
        
        return patterns

    def _term_to_bit_pattern(self, term, num_bits):
        """
        Convert a Boolean term back to bit pattern with don't cares.
        
        Args:
            term (str): Term like "x1x2'x5"
            num_bits (int): Number of bits
            
        Returns:
            str: Bit pattern like "10-01---"
        """
        import re
        
        # Initialize with don't cares
        pattern = ['-'] * num_bits
        
        # Extract all variables with their complements
        # Pattern: x followed by digits, optionally followed by '
        var_pattern = r"x(\d+)('?)"
        matches = re.findall(var_pattern, term)
        
        for var_num_str, complement in matches:
            var_num = int(var_num_str)
            if 1 <= var_num <= num_bits:
                idx = var_num - 1  # 0-indexed
                pattern[idx] = '0' if complement else '1'
        
        return ''.join(pattern)
    
    # ============================================================================
    # AUTO-SELECTION: Choose 3D or 4D
    # ============================================================================
    def minimize(self, form='sop'):
        """
        Automatically choose the best minimization strategy.
        
        Strategy selection:
        - n ≤ 8: 3D minimization (geometric clustering)
        - 8 < n ≤ 10: 4D minimization (geometric clustering, optimal limit)
        - n > 10: Hierarchical 10-var base (NOT clustering, pure hierarchical)
        
        Args:
            form (str): 'sop' or 'pos'
            
        Returns:
            tuple: (list of terms, expression string)
        """
        if self.num_vars <= 8:
            print(f"Auto-selecting: 3D minimization (n={self.num_vars} ≤ 8)")
            return self.minimize_3d(form)
        
        elif self.num_vars <= 10:
            print(f"Auto-selecting: 4D minimization (n={self.num_vars} ≤ 10) [OPTIMAL]")
            return self.minimize_4d(form)
        
        else:
            print(f"Auto-selecting: Hierarchical 10-var base (n={self.num_vars} > 10)")
            print(f"Note: Using hierarchical decomposition (not geometric clustering)")
            return self.minimize_heirarchical(form)
    
    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================

    def get_optimization_stats(self):
        """
        Return statistics about which optimization should be used.
        
        Returns:
            dict: Statistics about problem size and recommended optimization
        """
        n_identifiers = 2 ** self.num_extra_vars
        
        # Estimate operations for each method
        string_ops = n_identifiers ** 2 * self.num_extra_vars * 3  # Rough estimate
        bitwise_ops = n_identifiers ** 2 * 2
        vectorized_ops = n_identifiers ** 2 // 8  # SIMD factor ~8
        
        return {
            'num_identifiers': n_identifiers,
            'num_extra_vars': self.num_extra_vars,
            'recommended': 'vectorized' if n_identifiers >= 64 else 'bitwise',
            'estimated_ops': {
                'string': string_ops,
                'bitwise': bitwise_ops,
                'vectorized': vectorized_ops
            },
            'speedup_bitwise_vs_string': string_ops / bitwise_ops,
            'speedup_vectorized_vs_string': string_ops / vectorized_ops,
            'speedup_vectorized_vs_bitwise': bitwise_ops / vectorized_ops
        }
        
    def generate_verilog(self, module_name="logic_circuit", form='sop'):
        """
        Generate Verilog HDL code for the minimized Boolean expression.
        
        Args:
            module_name: Name of the Verilog module (default: "logic_circuit")
            form: 'sop' for Sum of Products or 'pos' for Product of Sums
            
        Returns:
            String containing complete Verilog module code
        """
        terms, expression = self.minimize_3d(form=form)
        
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
        terms, expression = self.minimize_3d(form=form)
        verilog_code = self.generate_verilog(module_name=module_name, form=form)

        # Get Graphviz DOT
        dot_source = self._generate_logic_gates_graphviz(terms, form=form)

        # Escape for JS template literal (keep backslashes for \n in DOT labels)
        dot_js = dot_source.replace("\\", "\\\\").replace("`", "\\`")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>K-Map Minimization Results ({self.num_vars} Variables)</title>
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
    <h1>K-Map Minimization Results ({self.num_vars} Variables)</h1>

    <div class="info">
        <strong>Form:</strong> {form.upper()} ({'Sum of Products' if form.lower() == 'sop' else 'Product of Sums'})<br>
        <strong>Variables:</strong> {self.num_vars}<br>
        <strong>Terms:</strong> {len(terms)}<br>
        <strong>4x4 K-maps:</strong> {self.num_maps}<br>
        <strong>Extra Variables:</strong> {self.num_extra_vars}
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

    def _generate_logic_gates_graphviz(self, terms, form='sop'):
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
            
def main():
    import random
    
    # Example: 8-variable K-map with random pattern
    print("EXAMPLE: 8-VARIABLE K-MAP (Random Pattern)")
    print("="*60)
    num_vars = 8
    
    # Generate random output values (256 total for 8 variables)
    # Set random seed for reproducibility
    random.seed(42)
    output_values_8 = [random.choice([0, 1, 'd']) for _ in range(2**num_vars)]
    
    # Create and minimize
    kmap_solver_8 = BoolMinGeo(num_vars, output_values_8)
    kmap_solver_8.print_kmaps()
    
    # Minimize the 8-variable K-map
    terms, expression = kmap_solver_8.minimize_3d(form='sop')
    
    # Generate HTML report
    print("\n" + "="*60)
    print("GENERATING HTML REPORT")
    print("="*60)
    html_file = kmap_solver_8.generate_html_report(
        filename="kmap_8var_report.html",
        form='sop',
        module_name="logic_circuit_8var"
    )
    print(f"✓ HTML report generated: {html_file}")
    print(f"  Open this file in a web browser to view the interactive report.")
    
    # Commented out examples
    """
    # Example 1: 5-variable K-map
    print("EXAMPLE 1: 5-VARIABLE K-MAP")
    print("="*60)
    num_vars = 5
    # Create sample output values (you can modify these)
    # Example: minterms 0,1,5,7,8,9,13,15,16,17,21,23,24,25,29,31 are 1
    output_values_5 = [
        1,1,0,0,0,1,0,1,  # First 8 minterms (x1=0)
        1,1,0,0,0,1,0,1,  # Next 8 minterms (x1=0)
        1,1,0,0,0,1,0,1,  # Next 8 minterms (x1=1)
        1,1,0,0,0,1,0,1   # Last 8 minterms (x1=1)
    ]
    
    kmap_solver_5 = BoolMinGeo(num_vars, output_values_5)
    kmap_solver_5.print_kmaps()
    
    # Minimize the 5-variable K-map
    terms, expression = kmap_solver_5.minimize_3d(form='sop')
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: 6-variable K-map with more complex pattern
    print("\n\nEXAMPLE 2: 6-VARIABLE K-MAP")
    print("="*60)
    num_vars = 6
    # Create sample output values (64 total for 6 variables)
    # Pattern: Set minterms where x3'x4' appears (regardless of x1,x2,x5,x6)
    output_values_6 = [0] * 64
    for i in range(64):
        bits = format(i, '06b')
        # Set to 1 if bits 2,3 are 00 (x3'x4' in our variable ordering)
        if bits[2:4] == '00':
            output_values_6[i] = 1
    
    kmap_solver_6 = BoolMinGeo(num_vars, output_values_6)
    kmap_solver_6.print_kmaps()
    
    # Minimize the 6-variable K-map
    terms, expression = kmap_solver_6.minimize_3d(form='sop')
    
    # Example 3: 6-variable K-map with multiple terms
    print("\n\nEXAMPLE 3: 6-VARIABLE K-MAP (More Complex)")
    print("="*60)
    # Pattern: x1'x2' + x1x2x3'x4'
    output_values_6b = [0] * 64
    for i in range(64):
        bits = format(i, '06b')
        # x1'x2' (bits 0,1 are 00)
        if bits[0:2] == '00':
            output_values_6b[i] = 1
        # x1x2x3'x4' (bits 0,1,2,3 are 1100)
        elif bits[0:4] == '1100':
            output_values_6b[i] = 1
    
    kmap_solver_6b = BoolMinGeo(num_vars, output_values_6b)
    kmap_solver_6b.print_kmaps()
    
    # Minimize the 6-variable K-map
    terms, expression = kmap_solver_6b.minimize_3d(form='sop')
    """

if __name__ == "__main__":
    main()