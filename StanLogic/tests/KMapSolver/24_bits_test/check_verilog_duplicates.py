"""
Check for duplicate terms in the Verilog file.

This script analyzes the boolean_function_24bit.v file and identifies:
1. Duplicate Verilog terms in the assign statement
2. Statistics about the expression
3. Potential optimization opportunities

Author: Somtochukwu Stanislus Emeka-Onwuneme
Date: December 2025
"""

import os
import re
from collections import Counter

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
VERILOG_DIR = os.path.join(OUTPUTS_DIR, "verilog")
VERILOG_FILE = os.path.join(VERILOG_DIR, "boolean_function_24bit.v")

def extract_verilog_terms(verilog_file):
    """
    Extract all terms from the Verilog assign statement.
    
    Returns:
        list: All terms in the expression
    """
    print(f"Reading Verilog file: {verilog_file}")
    
    if not os.path.exists(verilog_file):
        print(f"Error: File not found: {verilog_file}")
        return None
    
    file_size_mb = os.path.getsize(verilog_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    try:
        with open(verilog_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File read successfully ({len(content):,} characters)")
        
        # Find the assign statement
        # Pattern: assign result = <expression>;
        assign_pattern = r'assign\s+result\s*=\s*(.*?);'
        match = re.search(assign_pattern, content, re.DOTALL)
        
        if not match:
            print("Error: Could not find 'assign result = ...' statement")
            return None
        
        expression = match.group(1).strip()
        print(f"Extracted expression: {len(expression):,} characters")
        
        # Split by OR operator (|) to get individual terms
        # Verilog terms are joined with ' | '
        terms = []
        
        # More robust splitting that handles newlines and whitespace
        raw_terms = re.split(r'\s*\|\s*', expression)
        
        for term in raw_terms:
            term = term.strip()
            if term:
                terms.append(term)
        
        print(f"Extracted {len(terms):,} terms from expression")
        return terms
        
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_duplicates(terms):
    """
    Analyze terms for duplicates and provide statistics.
    
    Args:
        terms: List of Verilog term strings
    """
    print(f"\n{'='*80}")
    print("DUPLICATE ANALYSIS")
    print(f"{'='*80}")
    
    # Count occurrences of each term
    term_counts = Counter(terms)
    
    total_terms = len(terms)
    unique_terms = len(term_counts)
    duplicate_terms = sum(1 for count in term_counts.values() if count > 1)
    total_duplicates = sum(count - 1 for count in term_counts.values() if count > 1)
    
    print(f"\nStatistics:")
    print(f"  Total terms:           {total_terms:,}")
    print(f"  Unique terms:          {unique_terms:,}")
    print(f"  Terms with duplicates: {duplicate_terms:,}")
    print(f"  Total duplicate count: {total_duplicates:,}")
    print(f"  Redundancy ratio:      {(total_duplicates / total_terms * 100):.2f}%")
    
    if duplicate_terms > 0:
        print(f"\n{'='*80}")
        print("DUPLICATE TERMS FOUND")
        print(f"{'='*80}")
        
        # Sort by count (most duplicated first)
        duplicates = [(term, count) for term, count in term_counts.items() if count > 1]
        duplicates.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 20 most duplicated terms
        print(f"\nTop {min(20, len(duplicates))} most duplicated terms:")
        print(f"{'Count':<10} {'Term':<70}")
        print("-" * 80)
        
        for term, count in duplicates[:20]:
            # Truncate long terms for display
            display_term = term if len(term) <= 65 else term[:62] + "..."
            print(f"{count:<10} {display_term}")
        
        if len(duplicates) > 20:
            print(f"\n... and {len(duplicates) - 20} more duplicated terms")
        
        # Calculate potential file size reduction
        # Each duplicate term wastes: len(term) + len(' | ') characters
        wasted_chars = sum((count - 1) * (len(term) + 3) for term, count in duplicates)
        wasted_mb = wasted_chars / (1024 * 1024)
        
        print(f"\nPotential Savings:")
        print(f"  Wasted characters:     {wasted_chars:,}")
        print(f"  Wasted space:          {wasted_mb:.2f} MB")
        print(f"  Optimized file would have {unique_terms:,} terms instead of {total_terms:,}")
        
        return duplicates
    else:
        print("\n✓ No duplicates found - expression is already optimized!")
        return []

def save_duplicate_report(duplicates, output_file):
    """Save detailed duplicate report to file."""
    if not duplicates:
        print("\nNo duplicates to report.")
        return
    
    print(f"\nSaving detailed report to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VERILOG DUPLICATE TERMS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"File analyzed: {VERILOG_FILE}\n")
        f.write(f"Total duplicated terms: {len(duplicates):,}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("COMPLETE DUPLICATE LIST\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"{'Count':<10} {'Term'}\n")
        f.write("="*80 + "\n")
        
        for term, count in duplicates:
            f.write(f"{count:<10} {term}\n")
    
    print(f"  Report saved: {output_file}")

def main():
    print("="*80)
    print("VERILOG DUPLICATE CHECKER")
    print("="*80)
    print()
    
    # Extract terms from Verilog file
    terms = extract_verilog_terms(VERILOG_FILE)
    
    if terms is None:
        print("\nFailed to extract terms from Verilog file")
        return False
    
    # Analyze for duplicates
    duplicates = analyze_duplicates(terms)
    
    # Save report if duplicates found
    if duplicates:
        report_file = os.path.join(OUTPUTS_DIR, "verilog_duplicates_report.txt")
        save_duplicate_report(duplicates, report_file)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
