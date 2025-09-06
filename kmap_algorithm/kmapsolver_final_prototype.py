# ------------------------------
# Flexible K-map solver prototype
# Supports: 2, 3, and 4 variables
# Handles don't-care conditions (d)
# ------------------------------

# --- USER SETUP --- #
num_vars = 4   # <-- Change this to 2, 3, or 4 as needed

if num_vars == 2:
    row_labels = ["0", "1"]      # x2
    col_labels = ["0", "1"]      # x1
    kmap = [
        [1, 0],
        [1, 1]
    ]
elif num_vars == 3:
    row_labels = ["0", "1"]      # x3
    col_labels = ["00", "01", "11", "10"]  # x1x2 (Gray-coded)
    kmap = [
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ]
elif num_vars == 4:
    row_labels = ["00", "01", "11", "10"]  # x3x4 (Gray-coded)
    col_labels = ["00", "01", "11", "10"]  # x1x2 (Gray-coded)
    # Define your K-map (replace with your own map if needed)
    kmap = [
        [0, 1, 'd', 0],
        [0, 1, 'd', 0],
        [0, 0, 'd', 0],
        [1, 1, 'd', 1]
    ]
else:
    raise ValueError("Only 2, 3, or 4 variables are supported.")

#Determine row and column labels, and map size, map size must be rectangular
num_rows = len(row_labels)
num_cols = len(col_labels)

def print_kmap(kmap):
    # Print header row
    print("     " + "  ".join(col_labels))

    # Print each row with its label
    for i, row in enumerate(kmap):
        print(f"{row_labels[i]}   " + "  ".join(str(val) for val in row))

# Run it
print_kmap(kmap)

def cell_to_minterm(r, c):
    """Iterate through the index list of col_labels and row_labels and 
    concatentates the index results
    
    e.g if col_labels[] = 00 and row_labels[] = 11, 
    bits = 0011 
    
    Iteration ends only when all cells have been visited"""
    bits = col_labels[c] + row_labels[r]  # x1 x2 x3 x4

    """returns the the minterm index of the concatentation in the k_map,
    e.g 1111 is (m)15"""
    return int(bits, 2)

def cell_to_bits(r, c):
    #returns the bit representation of the index
    return col_labels[c] + row_labels[r]

def simplify_group_bits(bits_list):
    # Each cell in the K-map was converted to a bitstring by cell_to_minterm(),
    # e.g., col_labels[c] + row_labels[r] → "1011" for (x1=1, x2=0, x3=1, x4=1).
    # bits_list contains those strings for all cells in the current group.

    # Start with the bit pattern of the first cell in the group
    bits = list(bits_list[0])  

    # Compare against the remaining cells in the group
    for b in bits_list[1:]:  
        # Look at each of the input variables (x1, x2, x3, x4)
        for i in range(len(bits)):  
            # If this variable differs across any two cells, mark it as '-'
            # '-' means "don't care" → the variable is eliminated from the implicant
            #
            # Example:
            #   bits_list = ["1011", "1001"]  (two adjacent cells)
            #   Compare:
            #      x1: '1' vs '1' → keep '1'
            #      x2: '0' vs '0' → keep '0'
            #      x3: '1' vs '0' → mismatch → '-'
            #      x4: '1' vs '1' → keep '1'
            #   Result = "10-1" → x1 x2' x4
            if bits[i] != b[i]:
                bits[i] = '-'

    vars_ = ['x1', 'x2', 'x3', 'x4'][:num_vars]
    term = []

    # Translate the simplified bit pattern into a Boolean product term
    for i, b in enumerate(bits):
        if b == '0':
            term.append(vars_[i] + "'")  # '0' → variable appears complemented
        elif b == '1':
            term.append(vars_[i])        # '1' → variable appears uncomplemented
        # '-' → variable drops out completely (not added to the term)

    # Return the final simplified product term (SOP component)
    return "".join(term)

def get_group_coords(r, c, h, w):
    # Given a starting cell (r, c) and a group size (h x w),
    # this comprehension collects the coordinates of every cell in the rectangle.
    # The modulo (%) allows "wrapping around" the edges of the K-map
    # (so groups can span from one side to the other, as in Karnaugh maps).
    return [((r+i) % num_rows, (c+j) % num_cols) for i in range(h) for j in range(w)]

    # All possible rectangular group sizes in a 4-variable K-map.
    # Must be powers of two: 1, 2, or 4 cells in each dimension.
    # Example: (2,2) = block of 4 cells, (1,4) = full row, (4,4) = the entire map.

# group sizes depend on number of variables (grid shape)
if num_vars == 2:
    group_sizes = [(1,1),(1,2),(2,1),(2,2)]
elif num_vars == 3:
    group_sizes = [(1,1),(1,2),(2,1),(1,4),(2,2),(2,4)]
else:  # 4-variable
    group_sizes = [(1,1),(1,2),(2,1),(2,2),(1,4),(4,1),(2,4),(4,2),(4,4)]

def find_all_groups(kmap):
    groups = []

    # Try every possible starting cell in the K-map
    for r in range(num_rows):
        for c in range(num_cols):

            # Try every possible group size (height h, width w)
            # Groups must be rectangular and have dimensions that are powers of two
            for h, w in group_sizes:

                # Get all coordinates (rr, cc) of the cells inside this rectangle
                # Wrapping with % ensures groups can cross the map edges
                coords = get_group_coords(r, c, h, w) #forms all possible rectangular combinations

                # --- DON'T CARE SUPPORT ---
                # Accept group if *all* cells are either 1 or 'd'
                # But reject if the group is *only* don't cares with no 1
                values = [kmap[rr][cc] for rr, cc in coords]
                if all(val in (1, 'd') for val in values) and any(val == 1 for val in values):

                    # Check: ignore single isolated 'd' that can't help clustering:
                    # (If there's exactly 1 cell and it's 'd', skip it)
                    if not (len(coords) == 1 and values[0] == 'd'):
                        # Store this group as a frozenset of coordinates
                        # frozenset is used so duplicate groups (same cells found from
                        # different starting positions) can be removed later
                        groups.append(frozenset(coords)) #appends the coordinates of the groups of 1's that exist in the k-map
                        #frozenset is chosen to avoid duplicates in the set, the set is a 'nested' set

    # Remove duplicates by converting the list to a set, then back to a list
    #The list has the following framework 
    #     List
    #  ├── set
    #  │    ├── Tuple(row, col)
    #  │    ├── Tuple(row, col)
    return list(set(groups)) 

def filter_prime_implicants(groups):
    primes = []
    for g in groups:
        #For each index in the list 'groups', iterate through the sets in each index and remove duplicates to ensure only prime implicants are obtained
        # A group is "prime" if it is not fully contained within another larger group.

        """
        (g < other) for other in groups if other != g asks to check if the current set 
        is a subset of other sets in 'groups' excluding the current indexed set
        # Example:
        #   g1 = {(0,0), (0,1)}   (size 2)
        #   g2 = {(0,0)}          (size 1)
        # g2 is redundant, since g1 already covers (0,0).
        then primes.append(g) returns g1
        """
        if not any((g < other) for other in groups if other != g):
            primes.append(g)
    return primes


# --- MAIN PIPELINE --- #

# Step 1: Find all valid groups of 1’s in the K-map
all_groups = find_all_groups(kmap)

# Step 2: Reduce to only prime implicants (maximal groups)
prime_groups = filter_prime_implicants(all_groups)

# Step 3: For each prime group, compute:
#   - the list of minterms it covers
#   - the simplified Boolean expression for that group
prime_terms = []   # Boolean expression strings (e.g., x1x3')
prime_covers = []  # Lists of minterm indices (e.g., [2,3,10,11])

for g in prime_groups:
    coords = sorted(g)  # all (row,col) cells in this group

    # Only 1s contribute to coverage; 'd' (don't care) helps form groups but doesn't require coverage.
    minterms = sorted(cell_to_minterm(r, c) for r, c in coords if kmap[r][c] == 1)

    bits_list = [cell_to_bits(r, c) for r, c in coords]
    term_str = simplify_group_bits(bits_list)

    prime_terms.append(term_str)
    prime_covers.append(minterms)

# Step 4: Build the prime implicant chart
#   - Map each minterm → the prime implicants that cover it
from collections import defaultdict 
#defaultdict allows us to create a new empty list by default 
minterm_to_primes = defaultdict(list)
all_minterms = set()

for idx, cover in enumerate(prime_covers):
    # prime_covers is a list, where each element is a list of 
    # minterms covered by one prime implicant.
    # prime_covers = [
    # [0,1,2,3],   # prime implicant (idx) 0 covers minterms 0–3
    # [4,6],       # prime implicant 1 covers minterms 4 and 6
    # [5,7]        # prime implicant 2 covers minterms 5 and 7
    # ]
    for m in cover:
        minterm_to_primes[m].append(idx) #maps minterms to primes
        # Example output
        # minterm_to_primes = {
        # 0: [0],
        # 1: [0],
        # 2: [0, 1],
        # 3: [0],
        # 4: [1],
        # 6: [1],
        # 5: [2],
        # 7: [2]
        # }
        all_minterms.add(m) # collect all required minterms

# Step 5: Identify essential primes
# A prime implicant is essential if it covers a minterm
# that no other prime covers.
essential_indices = set()
for m, primes in minterm_to_primes.items(): #for each m, check the list of primes that cover it
    if len(primes) == 1:
        essential_indices.add(primes[0])

# Get the Boolean terms of those essential primes
# eg if the essential indices in sorted(essential_indices are 0,2), 
# it prints the prime_terms correcsponding to that index
essential_terms = [prime_terms[i] for i in sorted(essential_indices)]

# Step 6: Mark minterms already covered by essentials
covered_minterms = set()
for i in essential_indices:
    covered_minterms.update(prime_covers[i]) # you have a list of prime covers, update the prime covers that 
    # correspond to the index 'i' in essential indices

# Step 7: If some minterms remain uncovered, select additional primes.
# We use a greedy heuristic:
#   At each step, choose the prime that covers the most uncovered minterms.
remaining = set(all_minterms) - covered_minterms
selected = set(essential_indices)

# --- Step 7: Cover remaining minterms with a greedy heuristic ---
while remaining:  # keep going until all minterms are covered
    best_idx = None
    best_cover_count = -1

    # Try each prime implicant and see how many *new* minterms it covers
    for idx in range(len(prime_groups)):
        if idx in selected: 
            continue  # skip primes we’ve already picked (essentials + earlier choices)

        # Intersection: minterms in this prime covers ∩ uncovered minterms
        cover = set(prime_covers[idx]) & remaining  

        # Example:
        # remaining = {3,4,5}
        # prime_covers[1] = [3,4]  → cover = {3,4}
        # prime_covers[2] = [4,5]  → cover = {4,5}

        # Pick the prime that covers the *most* new minterms
        if len(cover) > best_cover_count:
            best_cover_count = len(cover)
            best_idx = idx

    if best_idx is None:  # edge case: no prime can cover remaining minterms
        break

    # Add this prime to our selection
    selected.add(best_idx)

    # Mark all minterms this prime covers as "covered"
    covered_minterms.update(prime_covers[best_idx])

    # Recompute uncovered minterms
    remaining = set(all_minterms) - covered_minterms

    # Example continuation:
    # First loop → pick prime [3,4] (covers 2 minterms)
    # covered_minterms = {0,1,2,3,4}, remaining = {5}
    # Second loop → pick prime [4,5], now remaining = ∅ → done

# Step 8: Final minimized SOP = all selected prime terms
final_terms = [prime_terms[i] for i in sorted(selected)]

# Finally, remove redundant terms (test each for redundancy)
# --- Helper function ---
def covers_with_terms(chosen_indices):
    """
    Given a set of prime implicant indices,
    return the set of all minterms they cover.
    """
    covered = set()
    for idx in chosen_indices:
        covered.update(prime_covers[idx])  
        # Example:
        # if prime_covers[0] = [0,1,2,3]
        # and prime_covers[2] = [4,5],
        # then chosen_indices = {0,2} → covered = {0,1,2,3,4,5}
    return covered

# --- Step: Remove redundant primes ---
chosen = set(sorted(selected))   # start with all selected primes
# Example: selected = {0,1,2}

for idx in sorted(list(chosen)): # test each prime one at a time
    trial = chosen - {idx}       # temporarily drop this prime
    # Example: if chosen = {0,1,2} and idx = 1 → trial = {0,2}

    # Check if dropping this prime changes coverage
    if covers_with_terms(trial) == covers_with_terms(chosen):
        # If coverage is identical, then this prime was redundant.
        chosen = trial
        # Example:
        # chosen = {0,1,2}
        # trial = {0,2}
        # if {0,2} covers the same minterms as {0,1,2},
        # then prime 1 is redundant → update chosen = {0,2}

final_terms = [prime_terms[i] for i in sorted(chosen)] 
# prints the prime terms (x1,x2 etc) that correspond to the 
# non-redundant minterms

# Output
print("\nPrime implicants (term -> minterms):")
for t, cov in zip(prime_terms, prime_covers):
    print(f"  {t:12s} -> {cov}")

print("\nEssential primes:", essential_terms)
print("Final minimal SOP terms:", final_terms)
print("Final minimal SOP (sum):", " + ".join(final_terms))