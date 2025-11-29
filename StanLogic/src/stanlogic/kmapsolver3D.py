from stanlogic import KMapSolver
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
        solver = KMapSolver(kmap_values, convention="vranseic")
        
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