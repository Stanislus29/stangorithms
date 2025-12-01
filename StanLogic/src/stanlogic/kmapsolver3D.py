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
    kmap_solver_8 = KMapSolver3D(num_vars, output_values_8)
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
    """

if __name__ == "__main__":
    main()