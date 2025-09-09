# A Computational Framework for Minimizing SOP Expressions in 2–4 Variable Karnaugh Maps

Author: Somtochukwu Stanislus Emeka-Onwuneme

Publication Date: 8th August, 2025

---

## Table of Contents 

1. [Abstract](#1-abstract)  
2. [Introduction](#2-introduction)  
   - [The Sum-of-Products (SOP)](#21-the-sum-of-products-sop)  
   - [Karnaugh Maps](#22-karnaugh-maps) 
   - [Don’t Care conditions](#23-dont-care-conditions) 
3. [Methodology](#3-methodology)  
   - [Breakdown of the Algorithm](#31-breakdown-of-the-algorithm)  
   - [Pseudocode](#32-pseudocode)  
   - [Flowchart](#33-flowchart)  
4. [Using the Framework](#4-using-the-framework)  
   - [Using the Library](#41-using-the-library)  
   - [Entity-Relationship of class KMapSolver](#42-entity-relationship-of-class-kmapsolver)  
5. [Conclusion](#5-conclusion) 
    - [Complexity Analysis](#51-complexity-analysis)
    - [Comparison to Manual K-maps](#52-comparison-to-manual-k-maps)
    - [Limitations & Future Work](#53-limitations--future-work)
    - [Final Notes](#54-final-notes)
6. [References](#6-references)  


## 1. Abstract

This project presents a computational framework for simplifying Boolean functions using Karnaugh Maps (K-maps) with support for 2-, 3-, and 4-variable systems. The algorithm automates the process of identifying prime implicants, essential prime implicants, and redundant terms to produce the minimal Sum-of-Products (SOP) expression. In addition, the solver supports don’t care conditions, allowing further optimization when certain input combinations can be ignored.

The library is implemented in Python using only standard libraries, making it lightweight and easy to integrate into digital design workflows. It provides a structured pipeline — from grouping adjacent minterms in the K-map, through prime implicant filtering, to constructing the final minimal expression. By automating this process, the solver eliminates the tedium and potential errors of manual K-map simplification, while preserving the fundamental principles taught in digital logic design.

This tool is intended for students, educators, and researchers in computer engineering, electrical engineering, and computer science who need a practical aid for circuit minimization, as well as developers seeking to integrate K-map simplification into broader EDA (Electronic Design Automation) tools.

## 2. Introduction

This folder contains an algorithmic framework for obtaining the minimal SOP expression for n-variable k-maps (Karnaugh maps) for 1< n <= 4. 

K-maps are used in the design of digital systems to obtain expressions that implement a digital circuit at the lowest cost. The cost of a digital circuit is determined by it's inputs and the number of gates in it's boolean expression.

I recommend that anyone not well versed in boolean algebra, and the manual simplification of kmaps look at the material in the references, before attempting to understand this computational model. 

Generally, kmaps are simplified based off the framework provided by the **Quince-McCluskey** algorithm, particularly for systems with n > 4.

The process of minimizing a truth table according to the Quince-McCluskey algorithm is in two phases:


**1. Prime Implicant Generation**

- List all minterms in binary form.

- Group them by the number of 1’s in their binary representation.

- Iteratively combine terms that differ by only one bit, replacing the differing bit with a “–” (don’t care).

- Continue until no further combinations are possible.

- The terms that cannot be combined further are the prime implicants.

**2. Prime Implicant Chart**

- Create a chart mapping minterms to the prime implicants that cover them.

- Identify essential prime implicants (those that uniquely cover at least one minterm).

- Use essential primes first, then select additional primes (often via Petrick’s method) to cover remaining minterms.

**Example**

Consider the function: 

f(a,b,c,d)=∑m(0,1,2,5,6,7,8,9,10,14)

**Step 1: Binary grouping**
```yaml
Group 0 (0 ones): 0000 (m0)
Group 1 (1 one): 0001 (m1), 0010 (m2), 1000 (m8)
Group 2 (2 ones): 0101 (m5), 0110 (m6), 1001 (m9), 1010 (m10)
Group 3 (3 ones): 0111 (m7), 1110 (m14)
```

**Step 2: First Level Combining**
```yaml
0000 & 0001 → 000-  
0000 & 0010 → 00-0  
0000 & 1000 → -000  
0001 & 0010 → 00-1  
1000 & 1001 → 100-  
...
```
- Continue until prime implicants are obtained.

**Step 3: Prime implicant chart**
Construct a table mapping each minterm to the implicants that cover it.

**Step 4: Select essential primes**
Cover all minterms with the minimal set of implicants.

---

Generally, the computational framework described in this documentation uses the principles specified by the Quince-McCluskey algorithm. However, conscious effort was taken to structure this program which implements the algorithm in a way as which a learner of digital systems might be taught to minimize functions from k-maps in textbooks. 

### 2.1 The Sum-of-Products (SOP)
The SOP (sum-of-products) expression is obtained by considering the minterms i.e the occurences in the k-map where the output is true. For that particular output, we set all input variables to be true. 

e.g if m16 (minterm 16) = 1, the inputs x1,x2....xN nominally equate to 1. However, if any of the variables in the bitstring does not equate to one, it is represented as a complement.

**Example**
The function $( f(x_1, x_2, x_3) $) is defined by the following truth table:  

| Row | $(x_1$) | $(x_2$) | $(x_3$) | $(f$) | Minterm       |
|-----|---------|---------|---------|-------|---------------|
| 0   | 0       | 0       | 0       | 0     | –             |
| 1   | 0       | 0       | 1       | 1     | $(m_1 = x_1'x_2'x_3$) |
| 2   | 0       | 1       | 0       | 0     | –             |
| 3   | 0       | 1       | 1       | 0     | –             |
| 4   | 1       | 0       | 0       | 1     | $(m_4 = x_1x_2'x_3'$) |
| 5   | 1       | 0       | 1       | 1     | $(m_5 = x_1x_2'x_3$)  |
| 6   | 1       | 1       | 0       | 1     | $(m_6 = x_1x_2x_3'$)  |
| 7   | 1       | 1       | 1       | 0     | –             |

---

**Step 1: Canonical SOP** 

Collect all rows where $( f = 1 $):  

$[
f(x_1, x_2, x_3) = m_1 + m_4 + m_5 + m_6
$]  

Substitute the minterms:  

$[
f(x_1, x_2, x_3) = x_1'x_2'x_3 + x_1x_2'x_3' + x_1x_2'x_3 + x_1x_2x_3'
$]  

---

**Step 2: Simplification**

Factor and reduce using boolean properties:  

$[
f = (x_1' + x_1)x_2'x_3 + x_1(x_2' + x_2)x_3'
$]  

$[
f = (1)x_2'x_3 + x_1(1)x_3'
$]  

$[
f = x_2'x_3 + x_1x_3'
$]  

---

The final minimal SOP is:  

$[
f(x_1, x_2, x_3) = x_2'x_3 + x_1x_3'
$]  

---

However, we see that with this approach to minimization, there is the issue of applying boolean properties like the distributive property to simplify and obtain the minimal cost expression. This can be very cumbersome. 

### 2.2 Karnaugh Maps

The Karnaugh Map provides an easy method for simplification. 

For the 3-variable truth table above, the k-map below shows the location of the minterms:

```yaml
        x₁x₂
      00   01   11   10
    +----+----+----+----+
x₃=0| m0 | m2 | m6 | m4 |
    +----+----+----+----+
x₃=1| m1 | m3 | m7 | m5 |
    +----+----+----+----+
```    

For our function above, the k-map representation thus is: 

```yaml
        x₁x₂
      00   01   11   10
    +----+----+----+----+
x₃=0|  0 |  0 |  1 |  1 |
    +----+----+----+----+
x₃=1|  1 |  0 |  0 |  1 |
    +----+----+----+----+
```    

To obtain the SOP, we find the clusters of 1's in the k-map.

The rules for clustering are: 
- Groups must be rectangular, of size 1,2,4 
- Group's must contain only 1's (don't cares if present, can also join)
- Groups can warp around edges 

**Clusters formed:**

1. **Cluster A:** (m₄, m₅) → $(x_1x_2'$)  
2. **Cluster B:** (m₄, m₆) → $(x_1x_3'$) 
3. **Cluster C:** (m₁, m₅) → $(x_2'x_3$)  

---

**Step 2: SOP Expression**

Now we combine all clusters:

$[
f(x_1,x_2,x_3) = x_1x_2' + x_1x_3' + x_2'x_3
$]

---

**Step 3: Minimality**

- Every **1** in the K-map is covered.  
- No term can be removed without leaving an uncovered 1.  
- Therefore, this SOP is **minimal**.  

---

*For better understanding, the reader is advised to refer to the texts in the references below.*

### 2.3 Don't Care Conditions

In many digital systems, certain input combinations never occur or their output value does not affect the behavior of the circuit. These cases are known as don’t cares and are typically denoted by ‘d’ in a K-map.

Don’t cares are useful because they can be flexibly treated as either 0 or 1 during the simplification process. This flexibility allows the algorithm to form larger rectangular clusters in the K-map, which leads to simpler SOP expressions and thus lower-cost digital circuits.

For example:

Consider the following 4-variable K-map:
```yaml
        x₁x₂
      00   01   11   10
    +----+----+----+----+
x₃x₄=00|  0 |  1 |  d |  0 |
    +----+----+----+----+
x₃x₄=01|  0 |  1 |  d |  0 |
    +----+----+----+----+
x₃x₄=11|  0 |  0 |  d |  0 |
    +----+----+----+----+
x₃x₄=10|  1 |  1 |  d |  1 |
    +----+----+----+----+
```
- The 1’s represent minterms where the function is true.

- The ‘d’ values are don’t cares.

**Step 1: Normal grouping without don’t cares**

If we ignored the don’t cares, we would be limited to smaller groups, leading to a more complex SOP.

**Step 2: Using don’t cares in grouping**

By treating the don’t cares as 1’s, we can form larger rectangles:

- A group of 4 using the cells in columns 01 and 11.

- Another group of 4 along the bottom row, wrapping across edges.

This reduces the number of product terms significantly.

**Step 3: Resulting SOP**
The simplified expression (for this example) would look like:
$[f(x_1​,x_2​,x_3​,x4​)=x_2​ + x_3​x_4′$]​

Thus, don't cares allow us to merge more cells. leading to smaller and cheaper logic implementations. 

## 3. Methodology

The algorithm combines various data structures: lists, sets and dictionaries to solve the problem. Nested loops are also a feature of this algorithm. It should be noted that the algorithm was developed after an initial study of the Quine-McCluskey algorithm, and seeks to provide a computational framework to achieving the same objective.

The computer program was written in Python and developed in three steps: 
1. An initial 4 variable k-map solver algorithm was developed
2. The algorithm was expanded to handle don't care's and 2, and 3 variable k-maps
3. The algorithm was further expanded into a library for easy reuse when developing complex systems 

Any user of the algorithm who wishes to understand the computer program is advised to refer to ```kmapsolver_final_prototype.py```. The program is rich with comments and visual examples to help the reader understand what the program seeks to achieve. 

### 3.1 BREAKDOWN OF THE ALGORITHM 

1. The initial block of the program requires the user to determine the number of variables in the k-map. 

The variable ```kmap``` is a nested list (array), and the kmap which is stored in ```kmap``` depends on the number of variables chosen by the user.

```python 
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
```

2. It was considered necessary to print the kmap inputed by the user once the program has been run, this block handles that. 

```python
def print_kmap(kmap):
    # Print header row
    print("     " + "  ".join(col_labels))

    # Print each row with its label
    for i, row in enumerate(kmap):
        print(f"{row_labels[i]}   " + "  ".join(str(val) for val in row))

# Run it
print_kmap(kmap)
```

3. ```cell_to_minterm()``` performs the function of concatentating the row labels and column labels into a singular list - bits. The index of a bit string determines what minterm it covers. 

e.g 1111 is m15

```python
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
```
4. ```cell_to_bits()``` returns the bit representation of a cell

```python 
def cell_to_bits(r, c):
    #returns the bit representation of the index
    return col_labels[c] + row_labels[r]
```

5. This block inputs the bit strings from ```bits``` into a list, and then turns each bit_list into a string of 1's and 0's. 

The block also checks for the unchanging variables between two adjacent cells, if the variables change between adjacent cells, it is discared as an ignore '-', and the variable is eliminated from the implicant. 

It returns a list of terms, which are the product terms of the non-changing adjacent cells.

```python 
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
            # '-' means "ignore" → the variable is eliminated from the implicant
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
```

6. At this point, we have to find all possible clusters. The return function includes a modulo (%) for warping round the edges 

We also added an if - elif loop to determine the possible group sizes depedning on the number of variables decided at the start

```python
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
```

7. The program starts to get tricky here, so we take more detail in each line

The block below essentially says
"For each row, search each cell (index) in the row, and at each cell, form all possible rectangular combinations."

```python
def find_all_groups(kmap):
    groups = []

    # Try every possible starting cell in the K-map
    for r in range(num_rows):
        for c in range(num_cols):

            # Try every possible group size (height h, width w)
            # Groups must be rectangular and have dimensions that are powers of two
            for h, w in group_sizes:
                  coords = get_group_coords(r, c, h, w) #forms all possible rectangular combinations
```                

By calling the function ```get_group_coords(r,c,h,w)```, we activate the function which returns a list which has a tuple as it's subset. 

```python
return [((r+i) % num_rows, (c+j) % num_cols) for i in range(h) for j in range(w)]
```

The modulo in the return statement above allows wrapping round edges which is crucial in solving k-maps.

e.g if get_group_coords(1, 1, 2, 2)
That means: start at cell (1,1), make a 2×2 block.

Output:
```python
[(1,1), (1,2),
 (2,1), (2,2)]
```

Another way of looking at it is to say, at a cell, move horizontally by the value of w, and vertically by the value of h, where h and w are the height and width of all possible rectangles stored in tuples within the list ```group_sizes```

The next block is self-explanatory:

```python   
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
    return list(set(groups))                 
```    

After tracing the clusters that contain only 1's from each cell and appending each group of coordinates into a set. 

```python 
groups.append(frozenset(coords))
```

groups looks like this

```yaml
 List
   ├── set of coordinates
    │    ├── Tuple(row, col)
    │    ├── Tuple(row, col)
   ├── set of coordinates
    │    ├── Tuple(row, col)
    │    ├── Tuple(row, col) 
```    

It is possible that the rectangular combinations of a cell may be the same as another, so to avoid duplicate coordinates, we convert the list to a set, and then back to a list 

```python
return list(set(groups))
```

8. This block iterates through the list ```groups```, and finds prime implicants, i.e coordinates which are not subset of other coordinates. 

The comments in the block are self explanatory: 

```python
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
```    

9. The main pipeline uses the already defined functions to obtain the prime terms and prime covers. 

```python
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
```

This block below is a bit tricky.

What we are doing is obtaining the bit_list of a cell in the k-map for each particular coordinate

The term_str uses the ```simplify_group_bits()``` to convert the bit_list into a variable map e.g. x1,x2,x3,x4

```python
    # Only 1s contribute to coverage; 'd' (don't care) helps form groups but doesn't require coverage.
     minterms = sorted(cell_to_minterm(r, c) for r, c in coords if kmap[r][c] == 1)
     
    bits_list = [cell_to_bits(r, c) for r, c in coords if kmap[r][c] == 1]

    term_str = simplify_group_bits(bits_list)

    prime_terms.append(term_str)
    prime_covers.append(minterms)
```

10. Each coordinate in coords is a prime implicant which covers a number of minterms. The minterms covered by a prime implicant is stored in prime_covers. 

*We see the importance of sorting at play here, by sorting the coordinates list and the minterm list, we ensure their indexes align*

The comments in the last portion of the block is self explanatory 

```python
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
```

### 3.2 PSEUDOCODE
Algorithm: K-map Minimization with Don't Care Support

INPUT: 
    num_vars → number of input variables (2, 3, or 4)
    kmap → grid of values (0, 1, or 'd')

OUTPUT:
    Minimal SOP expression

--------------------------------------------------
1. Initialize row_labels and col_labels depending on num_vars
2. Print the K-map (for user verification)

3. Define helper functions:
    - cell_to_minterm(r, c): map (row, col) to minterm index
    - cell_to_bits(r, c): return bitstring of a cell
    - simplify_group_bits(bits_list): eliminate changing vars, 
      keep only stable variables to form product term
    - get_group_coords(r, c, h, w): return coordinates for 
      rectangular group (supports wrapping across edges)

4. Set group_sizes depending on num_vars
   (e.g., for 4 vars: (1,1), (1,2), (2,1), (2,2), (1,4), (4,1), (2,4), (4,2), (4,4))

5. Find all valid groups:
    For each cell in the K-map:
        For each possible group size (h, w):
            coords ← get_group_coords(r, c, h, w)
            values ← kmap[rr][cc] for rr,cc in coords
            If all values are (1 or 'd') AND at least one is 1:
                If not (single isolated 'd'):
                    Add coords to groups

6. Filter prime implicants:
    A group is prime if it is not fully contained within another group

7. Build prime implicant data:
    For each prime group:
        minterms ← indices of all covered 1’s
        term_str ← simplify_group_bits(bitstrings)
        Store term_str and minterms

8. Construct prime implicant chart:
    Map each minterm → the prime implicants that cover it

9. Identify essential primes:
    Any prime covering a unique minterm is essential

10. Mark minterms covered by essentials

11. Cover remaining minterms (if any):
    While uncovered minterms exist:
        Select the prime that covers the most uncovered minterms
        Add it to selected set
        Update covered minterms

12. Remove redundant primes:
    For each chosen prime:
        If dropping it does not reduce coverage, remove it

13. Return result:
    Final minimal SOP expression =
        sum of all selected non-redundant prime terms

### 3.3 FLOWCHART 
```mermaid
flowchart TD

A[Start: Input num_vars and K-map] --> B[Initialize row_labels, col_labels, kmap]
B --> C[Print K-map]

C --> D[Find all possible groups]
D --> D1[Check group values: all 1 or d, at least one 1]
D1 --> D2{Valid group?}
D2 -- Yes --> E[Store group as frozenset(coords)]
D2 -- No --> D

E --> F[Filter prime implicants]
F --> G[Compute prime_terms and prime_covers]

G --> H[Build prime implicant chart]
H --> I[Identify essential primes]

I --> J[Mark covered minterms]
J --> K{Remaining uncovered minterms?}
K -- Yes --> L[Greedy select prime covering most uncovered]
L --> J
K -- No --> M[Remove redundant primes]

M --> N[Final SOP: Sum of selected prime terms]
N --> O[End]
```

## 4. USING THE FRAMEWORK

#### Installation and Usage

Clone the repositiory 

```bash 
git clone https://github.com/<your-username>/kmap-solver.git
cd kmap-solver
```

Run the solver: 
```bash
python kmapsolver_final_prototype.py
```
Modify the ```num_vars``` and ```kmap``` matrix inside the file to suit your input.

#### Example Run 

Input k-map:

```yaml 
        x₁x₂
      00   01   11   10
    +----+----+----+----+
x₃=0|  0 |  0 |  1 |  1 |
    +----+----+----+----+
x₃=1|  1 |  0 |  0 |  1 |
    +----+----+----+----+
```

Program Output: 

```text
Prime implicants (term -> minterms):
  x1x2'       -> [4, 5]
  x1x3'       -> [4, 6]
  x2'x3       -> [1, 5]

Essential primes: ['x1x2'']
Final minimal SOP terms: ['x1x2'', 'x1x3'', 'x2'x3']
Final minimal SOP (sum): x1x2' + x1x3' + x2'x3
```
### 4.1 Using the library
Within the repo, there is a library; ```lib_kmaap_solver.py```, the ```KMapSolver``` class provides a reusable way to compute minimal SOP expressions for 2,3, and 4 variable K-maps

You can initialize it with your K-map and run ```.minimize()``` to get the result.

**Example Run**
```python
from kmap_solver import KMapSolver

# Example: 3-variable K-map
kmap = [
    [0, 0, 1, 1],
    [1, 0, 0, 1]
]

solver = KMapSolver(kmap)

# Print the K-map for verification
solver.print_kmap()

# Get minimal SOP expression
terms, sop = solver.minimize()

print("\nMinimal SOP terms:", terms)
print("Minimal SOP expression:", sop)
```

Output 
```text
     00  01  11  10
0   0  0  1  1
1   1  0  0  1

Minimal SOP terms: ['x1x2'', 'x1x3'', "x2'x3"]
Minimal SOP expression: x1x2' + x1x3' + x2'x3
```

### 4.2 Entity-Relationship of class ```KMapSolver```

**Attributes**

```kmap``` → the input K-map (2D list of 0, 1, or 'd')

```num_rows``` → number of rows in the K-map

```num_cols``` → number of columns in the K-map

```num_vars``` → number of variables (2, 3, or 4, inferred from K-map size)

```row_labels / col_labels``` → Gray-coded labels used for mapping

```group_sizes``` → list of valid group dimensions (powers of two)

**Methods**

```__init__(kmap)``` → Initialize solver with a given K-map

```print_kmap()``` → Pretty-print the K-map with row/column headers

```find_all_groups(allow_dontcare=False)``` → Return all valid groups of 1’s (and optionally don’t-cares)

```filter_prime_implicants(groups)``` → Remove groups that are subsets of others (prime implicants only)

```_cell_to_minterm(r, c)``` → Internal: map a cell to its minterm index

```_cell_to_bits(r, c)`` → Internal: return bitstring representation of a cell

```_simplify_group_bits(bits_list)``` → Internal: simplify group into Boolean term

```_get_group_coords(r, c, h, w)``` → Internal: get wrapped coordinates for group of size h × w

```minimize()``` → Perform full minimization pipeline and return:

```final_terms``` → list of SOP terms (strings)

```sop``` → full SOP expression as a string

## 5. Conclusion 

### 5.1 Complexity Analysis

**Group finding**: 𝑂(𝑁⋅𝐺), where N = number of cells, G = possible group sizes.

**Prime filtering**: Subset checks between groups add another polynomial layer.

**Greedy covering**: Linear in the number of uncovered minterms.

The algorithm is efficient for 2–4 variables, but scales poorly for higher variables compared to Quine–McCluskey or Espresso. 

### 5.2 Comparison to Manual K-maps

- Manually solving K-maps requires visual clustering of 1’s.

- This algorithm automates that clustering using sets and coordinate grouping.

- It guarantees coverage of all 1’s while eliminating redundant terms.

### 5.3 Limitations & Future Work

- Supports only 2, 3, and 4 variable K-maps.

- For n>4, Quine–McCluskey or Espresso logic minimizer is recommended.

- Does not support Product-of-Sums Implementation

- Current solver is console-based. Future work:

    - Graphical visualization of clusters

    - Extension to higher variable systems

    - Integration into larger CAD tools

*It would be possible to use 3D maps for variables up to 5 and 6, but for variables higher than these, the algorithm becomes computationally expensive.* 

### 5.4 Final Notes

The algorithm implemented above produced accurate results upon testing, and the computational implementation satisfied all relevant rules assocaited with building minimal cost-expressions for k-maps.

## 6. References

1. Brown, S., & Vranesic, Z. (2014). Fundamentals of Digital Logic with Verilog Design. New York: McGraw-Hill.

2. Quine, W.V. (1955), A Way to Simplify Truth Functions.https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/way-to-simplify-truth-functions/7E2F9C4C9F69F5C742A69F6BAF59D7A0

3. McCluskey, E.J. (1956), Minimization of Boolean Functions.https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1956.tb03835.x

4. R. Rudell, Espresso Logic Minimizer.https://www2.eecs.berkeley.edu/Pubs/TechRpts/1987/CSD-87-290.pdf 

---
Copyright © 2025 Somtochukwu Stanislus Emeka-Onwuneme