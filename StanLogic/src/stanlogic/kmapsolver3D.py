from stanlogic.BoolMin2D import BoolMin2D
from collections import defaultdict

class KMapSolver3D:
    def __init__(self, num_vars, output_values=None):
        """
        Initialize KMapSolver3D with the number of variables.
        Generates truth table combinations and creates 3D K-map structure.
        
        Args:
            num_vars (int): Number of variables in the truth table (must be >= 5)
            output_values (list): Optional list of output values for each minterm
        """
        if num_vars < 5:
            raise ValueError("KMapSolver3D requires at least 5 variables. Use KMapSolver for 2-4 variables.")
        
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
        
        # Gray code sequences for 4x4 K-map lfabeling (last 4 variables)
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
            list: 4x4 matrix of values suitable for KMapSolver
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
        
        # Create a KMapSolver instance for this 4x4 map
        solver = BoolMin2D(kmap_values, convention="vranseic")
        
        # Get the essential prime implicants (internal computation)
        target_val = 0 if form.lower() == 'pos' else 1
        
        # OPTIMIZATION 1: Check for constant functions (all same value)
        # Collect all defined (non-don't-care) cells
        defined_cells = []
        for r in range(solver.num_rows):
            for c in range(solver.num_cols):
                if solver.kmap[r][c] != 'd':
                    defined_cells.append(solver.kmap[r][c])
        
        # Handle edge cases for constant functions
        if not defined_cells:  # All don't cares
            # Return empty result for all don't cares
            return {
                'bitmasks': [],
                'terms_bits': [],
                'extra_vars': extra_combo
            }
        
        # Check if all defined cells are 0 (no target values)
        if all(cell == (1 - target_val) for cell in defined_cells):
            # All zeros for SOP or all ones for POS: return empty
            return {
                'bitmasks': [],
                'terms_bits': [],
                'extra_vars': extra_combo
            }
        
        # Check if all defined cells equal target value
        if all(cell == target_val for cell in defined_cells):
            # All ones for SOP or all zeros for POS: return full coverage
            full_mask = 0
            for r in range(solver.num_rows):
                for c in range(solver.num_cols):
                    if solver.kmap[r][c] == target_val:
                        idx = solver._cell_index[r][c]
                        full_mask |= (1 << idx)
            # Return a single term covering everything (constant 1 for SOP, 0 for POS)
            return {
                'bitmasks': [full_mask],
                'terms_bits': ['----'],  # All don't-cares = constant 1/0
                'extra_vars': extra_combo
            }
        
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
            has_target = False  # Track if group covers any target cells
            
            temp = gmask
            while temp:
                low = temp & -temp
                idx = low.bit_length() - 1
                temp -= low
                
                if idx not in solver._index_to_rc:
                    continue  # Skip invalid indices
                
                r, c = solver._index_to_rc[idx]
                
                # OPTIMIZATION 2: Include in coverage AND bit pattern ONLY if cell has target value
                # Don't-care cells can be part of the group (for size) but should NOT
                # affect the term simplification!
                if solver.kmap[r][c] == target_val:
                    cover_mask |= 1 << idx
                    bits_list.append(solver._cell_bits[r][c])  # Only use target cell bits!
                    has_target = True
            
            # OPTIMIZATION 3: Validate that group covers at least one target cell
            # Groups with only don't-cares should be skipped
            if not has_target or cover_mask == 0:
                continue
            
            # OPTIMIZATION 4: Ensure bits_list is non-empty before simplification
            if not bits_list or len(bits_list) == 0:
                continue
            
            prime_covers.append(cover_mask)
            
            # Simplify to get bit pattern (with '-' for don't cares)
            simplified_bits = self._simplify_bits_only(bits_list)
            
            # OPTIMIZATION 5: Skip invalid/empty terms
            if not simplified_bits or simplified_bits.strip() == "":
                # Remove the last added cover if term is invalid
                prime_covers.pop()
                continue
                
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
        
        # OPTIMIZATION 6: Enhanced greedy set cover with exhaustive coverage validation
        covered_mask = 0
        for i in essential_indices:
            covered_mask |= prime_covers[i]
        
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
        
        # OPTIMIZATION 7: Exhaustive coverage verification with forced complete coverage
        # Build expected coverage mask from ALL cells with target value
        expected_coverage = 0
        for r in range(solver.num_rows):
            for c in range(solver.num_cols):
                if solver.kmap[r][c] == target_val:
                    idx = solver._cell_index[r][c]
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
        
        # OPTIMIZATION 8: Handle empty results properly
        if not chosen:
            # If no terms but we had target values, something went wrong
            # Return empty result (will be handled at higher level)
            if expected_coverage != 0:
                # This should not happen - return empty for safety
                return {
                    'bitmasks': [],
                    'terms_bits': [],
                    'extra_vars': extra_combo
                }
        
        # Return bitmasks and bit patterns for chosen essential primes
        result_bitmasks = [prime_covers[i] for i in sorted(chosen)]
        result_bits = [prime_terms_bits[i] for i in sorted(chosen)]
        
        return {
            'bitmasks': result_bitmasks,
            'terms_bits': result_bits,
            'extra_vars': extra_combo
        }
    
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
    
    def minimize_3d(self, form='sop'):
        """
        Minimize the entire 3D K-map with proper multi-map coverage tracking.
        
        Args:
            form (str): 'sop' or 'pos'
            
        Returns:
            tuple: (list of minimized terms, complete expression string)
        """
        print(f"\n{'='*60}")
        print(f"MINIMIZING {self.num_vars}-VARIABLE K-MAP")
        print(f"{'='*60}\n")
        
        target_val = 0 if form.lower() == 'pos' else 1
        
        # Build global minterm coverage map
        global_minterms = set()
        for i, val in enumerate(self.output_values):
            if val == target_val:
                global_minterms.add(i)
        
        if not global_minterms:
            print("No target minterms found.")
            return ([], "0") if form.lower() == 'sop' else ([], "1")
        
        # Step 1: Solve each K-map and collect ALL candidate terms
        all_candidate_terms = []
        
        for extra_combo in sorted(self.kmaps.keys()):
            print(f"Solving K-map for extra vars: {extra_combo}")
            result = self._solve_single_kmap(extra_combo, form)
            
            print(f"  Found {len(result['bitmasks'])} prime implicants")
            
            for i, term_bits in enumerate(result['terms_bits']):
                full_pattern = extra_combo + term_bits
                covered_minterms = self._get_covered_minterms(extra_combo, result['bitmasks'][i])
                
                if covered_minterms:
                    all_candidate_terms.append({
                        'pattern': full_pattern,
                        'extra_combo': extra_combo,
                        '4bit_pattern': term_bits,
                        'covered_minterms': covered_minterms
                    })
                    print(f"    Term: {extra_combo}|{term_bits} covers {len(covered_minterms)} minterms")
        
        if not all_candidate_terms:
            print("No valid terms found.")
            return ([], "0") if form.lower() == 'sop' else ([], "1")
        
        print(f"\n{'='*60}")
        print("SECOND-LEVEL MINIMIZATION ACROSS ALL K-MAPS")
        print(f"{'='*60}\n")
        
        # Step 2: Find essential terms
        covered = set()
        essential_terms = []
        
        minterm_to_terms = defaultdict(list)
        for idx, term in enumerate(all_candidate_terms):
            for minterm in term['covered_minterms']:
                minterm_to_terms[minterm].append(idx)
        
        for minterm, term_indices in minterm_to_terms.items():
            if len(term_indices) == 1:
                term_idx = term_indices[0]
                if term_idx not in essential_terms:
                    essential_terms.append(term_idx)
                    covered.update(all_candidate_terms[term_idx]['covered_minterms'])
        
        print(f"Found {len(essential_terms)} essential terms")
        
        # Step 3: Greedy set cover
        selected_terms = set(essential_terms)
        remaining = global_minterms - covered
        
        max_iterations = len(all_candidate_terms) * 2
        iteration = 0
        
        while remaining and iteration < max_iterations:
            iteration += 1
            best_idx = None
            best_count = 0
            
            for idx, term in enumerate(all_candidate_terms):
                if idx not in selected_terms:
                    new_coverage = len(term['covered_minterms'] & remaining)
                    if new_coverage > best_count:
                        best_count = new_coverage
                        best_idx = idx
            
            if best_idx is None or best_count == 0:
                for idx, term in enumerate(all_candidate_terms):
                    if idx not in selected_terms and term['covered_minterms'] & remaining:
                        selected_terms.add(idx)
                        covered.update(term['covered_minterms'])
                        remaining = global_minterms - covered
                        break
                else:
                    break
            else:
                selected_terms.add(best_idx)
                covered.update(all_candidate_terms[best_idx]['covered_minterms'])
                remaining = global_minterms - covered
        
        # Step 4: Try to combine terms with same 4-bit pattern
        pattern_to_terms = defaultdict(list)
        for idx in selected_terms:
            term = all_candidate_terms[idx]
            pattern_to_terms[term['4bit_pattern']].append((idx, term))
        
        final_terms = []
        processed_indices = set()
        
        print("\nCombining terms across K-maps...")
        
        for pattern, terms_with_idx in pattern_to_terms.items():
            if len(terms_with_idx) == 1:
                idx, term = terms_with_idx[0]
                if idx not in processed_indices:
                    term_str = self._bits_to_term(term['pattern'], form)
                    final_terms.append(term_str)
                    processed_indices.add(idx)
                    print(f"Single term: {term['pattern']} -> {term_str}")
            else:
                # Multiple terms with same 4-bit pattern
                indices = [idx for idx, _ in terms_with_idx]
                extra_combos = [term['extra_combo'] for _, term in terms_with_idx]
                
                # Try to recursively combine extra variable combinations
                combined_terms = self._combine_extra_var_terms(extra_combos, pattern, form)
                
                if combined_terms:
                    for term_str in combined_terms:
                        final_terms.append(term_str)
                    processed_indices.update(indices)
                    print(f"Combined {len(extra_combos)} terms with pattern {pattern}")
                    print(f"  Result: {combined_terms}")
                else:
                    # Cannot combine - keep separate
                    for idx, term in terms_with_idx:
                        if idx not in processed_indices:
                            term_str = self._bits_to_term(term['pattern'], form)
                            final_terms.append(term_str)
                            processed_indices.add(idx)
                            print(f"Separate: {term['pattern']} -> {term_str}")
        
        # Remove redundant terms
        final_terms = self._remove_redundant_terms(final_terms, form, global_minterms)
        
        # Step 5: Apply second-level Quine-McCluskey to further minimize across ALL terms
        print("\nApplying final Quine-McCluskey minimization...")
        final_terms = self._second_level_minimize(final_terms, form, global_minterms)
        
        # Build final expression
        join_operator = " * " if form.lower() == 'pos' else " + "
        final_expression = join_operator.join(final_terms) if final_terms else ("0" if form.lower() == 'sop' else "1")
        
        print(f"\n{'='*60}")
        print(f"FINAL MINIMIZED EXPRESSION ({form.upper()}):")
        print(f"{'='*60}")
        print(f"F = {final_expression}\n")
        
        return final_terms, final_expression
    
    def _get_covered_minterms(self, extra_combo, bitmask):
        """
        Get the global minterm indices covered by a term in a specific K-map.
        
        Args:
            extra_combo (str): Binary string for extra variables
            bitmask (int): Bitmask of covered cells in the 4x4 map
            
        Returns:
            set: Set of global minterm indices
        """
        covered = set()
        kmap = self.kmaps[extra_combo]
        
        # Extract minterms from bitmask
        temp = bitmask
        while temp:
            low = temp & -temp
            local_idx = low.bit_length() - 1
            temp -= low
            
            # Map local index to row, col in 4x4 map
            for r in range(4):
                for c in range(4):
                    cell = kmap[r][c]
                    cell_bits = self._cell_to_bits_4x4(r, c)
                    cell_local_idx = int(cell_bits, 2)
                    if cell_local_idx == local_idx:
                        covered.add(cell['minterm'])
                        break
        
        return covered
    
    def _cell_to_bits_4x4(self, r, c):
        """
        Convert 4x4 K-map cell coordinates to 4-bit pattern using Gray code.
        """
        col_bits = self.gray_code_4[c]
        row_bits = self.gray_code_4[r]
        return col_bits + row_bits
    
    def _can_combine_extra_vars(self, extra_combos):
        """
        Check if extra variable combinations form a valid power-of-2 group.
        """
        if len(extra_combos) <= 1:
            return False
        
        # Count must be power of 2
        if len(extra_combos) & (len(extra_combos) - 1) != 0:
            return False
        
        if not extra_combos:
            return False
            
        # Find varying bit positions
        varying_positions = []
        bits = list(extra_combos[0])
        for combo in extra_combos[1:]:
            for i in range(len(bits)):
                if bits[i] != combo[i] and i not in varying_positions:
                    varying_positions.append(i)
        
        # Must vary in exactly log2(count) positions
        expected_varying = len(extra_combos).bit_length() - 1
        return len(varying_positions) == expected_varying
    
    def _combine_extra_var_terms(self, extra_combos, pattern_4bit, form):
        """
        Recursively combine terms with same 4-bit pattern but different extra vars.
        Uses Quine-McCluskey algorithm on the extra variable portion.
        
        Returns list of combined term strings.
        """
        if len(extra_combos) == 1:
            full_pattern = extra_combos[0] + pattern_4bit
            return [self._bits_to_term(full_pattern, form)]
        
        # Apply Quine-McCluskey to extra variable combinations
        current_patterns = [list(combo) for combo in extra_combos]
        
        max_iterations = 10
        for iteration in range(max_iterations):
            combined = []
            used = [False] * len(current_patterns)
            
            # Try to combine pairs
            for i in range(len(current_patterns)):
                if used[i]:
                    continue
                for j in range(i + 1, len(current_patterns)):
                    if used[j]:
                        continue
                    
                    # Check if patterns differ in exactly one position
                    diff_count = 0
                    diff_pos = -1
                    for k in range(len(current_patterns[i])):
                        if current_patterns[i][k] != current_patterns[j][k]:
                            if current_patterns[i][k] == '-' or current_patterns[j][k] == '-':
                                # Already don't-care in one - can't combine
                                diff_count = 999
                                break
                            diff_count += 1
                            diff_pos = k
                    
                    if diff_count == 1:
                        # Can combine
                        new_pattern = current_patterns[i].copy()
                        new_pattern[diff_pos] = '-'
                        combined.append(new_pattern)
                        used[i] = True
                        used[j] = True
                        break
            
            # Add uncombined patterns
            for i, pattern in enumerate(current_patterns):
                if not used[i]:
                    combined.append(pattern)
            
            if len(combined) == len(current_patterns):
                # No further combination possible
                break
            
            current_patterns = combined
        
        # Convert patterns to terms
        result_terms = []
        for extra_pattern in current_patterns:
            full_pattern = ''.join(extra_pattern) + pattern_4bit
            term_str = self._bits_to_term(full_pattern, form)
            result_terms.append(term_str)
        
        return result_terms
    
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
    
    def _remove_redundant_terms(self, terms, form, global_minterms):
        """Remove redundant terms that don't add new coverage."""
        if len(terms) <= 1:
            return terms
        
        # Convert terms back to coverage
        term_coverage = []
        for term in terms:
            pattern = self._term_to_pattern(term)
            covered = self._pattern_to_minterms(pattern)
            term_coverage.append(covered & global_minterms)
        
        # Remove redundant terms
        non_redundant = []
        for i, term in enumerate(terms):
            # Check if removing this term loses coverage
            other_coverage = set()
            for j, cov in enumerate(term_coverage):
                if i != j:
                    other_coverage.update(cov)
            
            if not term_coverage[i].issubset(other_coverage):
                non_redundant.append(term)
        
        return non_redundant if non_redundant else terms
    
    def _term_to_pattern(self, term):
        """Convert a Boolean term back to a bit pattern."""
        pattern = ['-'] * self.num_vars
        
        # Parse the term
        i = 0
        while i < len(term):
            if term[i] == 'x':
                # Extract variable number
                j = i + 1
                while j < len(term) and term[j].isdigit():
                    j += 1
                var_num = int(term[i+1:j])
                
                # Check for complement
                if j < len(term) and term[j] == "'":
                    pattern[var_num - 1] = '0'
                    i = j + 1
                else:
                    pattern[var_num - 1] = '1'
                    i = j
            else:
                i += 1
        
        return ''.join(pattern)
    
    def _pattern_to_minterms(self, pattern):
        """Convert a bit pattern to the set of minterms it covers."""
        # Find don't-care positions
        dc_positions = [i for i, bit in enumerate(pattern) if bit == '-']
        
        # Generate all combinations
        covered = set()
        for i in range(2 ** len(dc_positions)):
            # Create a specific combination
            bits = list(pattern)
            for j, pos in enumerate(dc_positions):
                bits[pos] = str((i >> j) & 1)
            
            # Convert to minterm index
            minterm = int(''.join(bits), 2)
            covered.add(minterm)
        
        return covered
    
    def _second_level_minimize(self, terms, form, global_minterms):
        """
        Apply Quine-McCluskey style minimization to the final terms.
        Combines terms that differ in only one variable.
        """
        if len(terms) <= 1:
            return terms
        
        # Convert terms to patterns
        term_patterns = [self._term_to_pattern(term) for term in terms]
        
        # Iteratively combine terms using proper Quine-McCluskey
        changed = True
        iteration = 0
        max_iterations = 10
        
        while changed and iteration < max_iterations:
            iteration += 1
            changed = False
            new_patterns = []
            used = [False] * len(term_patterns)
            
            # Try to combine each pair of patterns
            for i in range(len(term_patterns)):
                if used[i]:
                    continue
                    
                for j in range(i + 1, len(term_patterns)):
                    if used[j]:
                        continue
                    
                    # Check if patterns differ in exactly one position
                    combined = self._try_combine_patterns(term_patterns[i], term_patterns[j])
                    if combined is not None:
                        # Verify combined pattern covers only valid minterms
                        combined_minterms = self._pattern_to_minterms(combined)
                        all_minterms = set(range(2**self.num_vars))
                        
                        # Check the combined pattern doesn't add unwanted minterms
                        pattern_i_minterms = self._pattern_to_minterms(term_patterns[i])
                        pattern_j_minterms = self._pattern_to_minterms(term_patterns[j])
                        expected_minterms = pattern_i_minterms | pattern_j_minterms
                        
                        if combined_minterms == expected_minterms:
                            new_patterns.append(combined)
                            used[i] = True
                            used[j] = True
                            changed = True
                            print(f"  Combined: {term_patterns[i]} + {term_patterns[j]} -> {combined}")
                            break
            
            # Add uncombined patterns
            for i, pattern in enumerate(term_patterns):
                if not used[i]:
                    new_patterns.append(pattern)
            
            if not changed:
                break
            
            term_patterns = new_patterns
        
        # Convert back to terms
        final_terms = [self._bits_to_term(pattern, form) for pattern in term_patterns]
        
        # Remove redundancies again
        return self._remove_redundant_terms(final_terms, form, global_minterms)
    
    def _try_combine_patterns(self, pattern1, pattern2):
        """
        Try to combine two patterns that differ in exactly one position.
        Returns combined pattern or None if cannot combine.
        """
        if len(pattern1) != len(pattern2):
            return None
        
        diff_count = 0
        diff_pos = -1
        
        for i in range(len(pattern1)):
            if pattern1[i] != pattern2[i]:
                # Both must be 0 or 1 (not '-') at differing position
                if pattern1[i] == '-' or pattern2[i] == '-':
                    return None
                diff_count += 1
                diff_pos = i
        
        # Must differ in exactly one position
        if diff_count != 1:
            return None
        
        # Create combined pattern with '-' at the differing position
        combined = list(pattern1)
        combined[diff_pos] = '-'
        return ''.join(combined)
    
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
            
def main():
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
    
    kmap_solver_5 = KMapSolver3D(num_vars, output_values_5)
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
    
    kmap_solver_6 = KMapSolver3D(num_vars, output_values_6)
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
    
    kmap_solver_6b = KMapSolver3D(num_vars, output_values_6b)
    kmap_solver_6b.print_kmaps()
    
    # Minimize the 6-variable K-map
    terms, expression = kmap_solver_6b.minimize_3d(form='sop')

if __name__ == "__main__":
    main()