"""
Test file for BoolMinGeo cluster coordinate mapping functions.

This demonstrates how to use the cluster analysis functions to understand
information density and pattern distribution in K-map space.
"""

import sys
sys.path.insert(0, '../src')

from stanlogic.BoolMinGeo import BoolMinGeo


def test_get_cluster_coordinates():
    """Test getting minterm bit patterns from a cluster pattern."""
    print("\n" + "="*70)
    print("TEST 1: get_cluster_coordinates()")
    print("="*70)
    
    # Create 5-variable K-map
    num_vars = 5
    output_values = [1,1,0,0,0,1,0,1] * 4  # Simple repeating pattern
    
    solver = BoolMinGeo(num_vars, output_values)
    
    # Test with a pattern containing don't-cares
    pattern = "0-001"  # 1 identifier bit + 4 K-map bits
    print(f"\nPattern: {pattern}")
    print(f"Don't-cares: {pattern.count('-')}")
    
    minterms = solver.get_cluster_coordinates(pattern)
    print(f"Covers {len(minterms)} minterms:")
    for mt in minterms:
        print(f"  {mt}")
    
    # Verify expansion is correct (should be 2^1 = 2 minterms)
    assert len(minterms) == 2, f"Expected 2 minterms, got {len(minterms)}"
    print("\n✓ Test passed!")


def test_cluster_density_map():
    """Test density mapping for overlapping patterns."""
    print("\n" + "="*70)
    print("TEST 2: get_cluster_density_map()")
    print("="*70)
    
    # Create 6-variable K-map
    num_vars = 6
    output_values = [1 if i % 3 == 0 else 0 for i in range(64)]
    
    solver = BoolMinGeo(num_vars, output_values)
    
    # Create overlapping patterns
    patterns = {
        "00--01",  # 4 minterms
        "0---01",  # 8 minterms (includes the 4 above)
        "01--01",  # 4 minterms (different region)
    }
    
    print(f"\nAnalyzing {len(patterns)} patterns:")
    for p in patterns:
        minterms = solver.get_cluster_coordinates(p)
        print(f"  {p} → {len(minterms)} minterms")
    
    density = solver.get_cluster_density_map(patterns)
    
    print(f"\nDensity Statistics:")
    print(f"  Total unique minterms: {density['total_minterms']}")
    print(f"  Max density (overlaps): {density['max_density']}")
    print(f"  Min density: {density['min_density']}")
    print(f"  Avg density: {density['avg_density']:.2f}")
    
    # Show some high-density minterms
    print(f"\nHigh-density minterms (covered by multiple patterns):")
    high_density = [(mt, count) for mt, count in density['minterms'].items() 
                    if count == density['max_density']][:5]
    for mt, count in high_density:
        print(f"  {mt}: covered by {count} patterns")
    
    assert density['max_density'] == 2, "Expected max density of 2"
    print("\n✓ Test passed!")


def test_3d_cluster_regions():
    """Test grouping minterms by identifier."""
    print("\n" + "="*70)
    print("TEST 3: get_3d_cluster_regions()")
    print("="*70)
    
    # Create 7-variable K-map
    num_vars = 7
    output_values = [1] * 128  # All 1s for simplicity
    
    solver = BoolMinGeo(num_vars, output_values)
    
    # Patterns spanning different identifiers
    patterns = {
        "001-001",  # In identifier 001
        "010-001",  # In identifier 010
        "01--001",  # Spans identifiers 010 and 011
    }
    
    print(f"\nAnalyzing {len(patterns)} patterns:")
    for p in patterns:
        print(f"  {p}")
    
    regions = solver.get_3d_cluster_regions(patterns)
    
    print(f"\nRegions (grouped by identifier):")
    for identifier, minterms in sorted(regions.items()):
        print(f"\n  Identifier {identifier}: {len(minterms)} minterms")
        # Show first 3 minterms in each region
        for mt in minterms[:3]:
            print(f"    {mt}")
        if len(minterms) > 3:
            print(f"    ... and {len(minterms)-3} more")
    
    # Should have at least 3 identifiers covered
    assert len(regions) >= 3, f"Expected at least 3 identifiers, got {len(regions)}"
    print("\n✓ Test passed!")


def test_visualize_cluster_coverage():
    """Test visualization of cluster coverage."""
    print("\n" + "="*70)
    print("TEST 4: visualize_cluster_coverage()")
    print("="*70)
    
    # Create 5-variable K-map
    num_vars = 5
    output_values = [1] * 32
    
    solver = BoolMinGeo(num_vars, output_values)
    
    # Pattern that creates a clear visual pattern
    pattern = "0-0-1"  # 4 minterms forming a pattern
    
    print(f"\nVisualizing pattern: {pattern}")
    solver.visualize_cluster_coverage(pattern, display=True)
    
    # Test non-display mode
    viz_text = solver.visualize_cluster_coverage(pattern, display=False)
    assert isinstance(viz_text, str), "Should return string when display=False"
    assert "■" in viz_text, "Visualization should contain filled cells"
    
    print("✓ Test passed!")


def test_practical_use_case():
    """Demonstrate practical use case: analyzing minimization results."""
    print("\n" + "="*70)
    print("TEST 5: Practical Use Case - Analyzing Minimization Results")
    print("="*70)
    
    # Create 6-variable K-map with specific pattern
    num_vars = 6
    # Create a pattern where certain regions are dense
    output_values = []
    for i in range(64):
        bits = format(i, '06b')
        # Set to 1 if first 2 bits are 00 or 11
        output_values.append(1 if bits[:2] in ['00', '11'] else 0)
    
    solver = BoolMinGeo(num_vars, output_values)
    
    # Perform minimization
    print("\nPerforming minimization...")
    terms, expression = solver.minimize_3d(form='sop')
    
    print(f"\nMinimized to {len(terms)} terms")
    
    # Analyze the patterns
    print("\n" + "-"*70)
    print("ANALYZING INFORMATION DENSITY")
    print("-"*70)
    
    # Get all patterns (convert terms back to patterns)
    patterns = []
    for term in terms:
        try:
            pattern = solver._term_to_pattern(term, num_vars)
            patterns.append(pattern)
        except:
            pass
    
    if patterns:
        print(f"\nFound {len(patterns)} patterns to analyze")
        
        # Density analysis
        density = solver.get_cluster_density_map(patterns)
        print(f"\nDensity Analysis:")
        print(f"  Total unique minterms covered: {density['total_minterms']}")
        print(f"  Maximum overlap: {density['max_density']} patterns")
        print(f"  Average overlap: {density['avg_density']:.2f} patterns")
        
        # Regional analysis
        regions = solver.get_3d_cluster_regions(patterns)
        print(f"\nRegional Distribution:")
        print(f"  Number of K-maps with coverage: {len(regions)}")
        
        # Show coverage per region
        for identifier, minterms in sorted(regions.items())[:5]:
            print(f"    Identifier {identifier}: {len(minterms)} minterms")
        
        # Visualize one interesting pattern
        if patterns:
            print(f"\nVisualizing first pattern: {patterns[0]}")
            solver.visualize_cluster_coverage(patterns[0], display=True)
    
    print("\n✓ Practical use case complete!")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*70)
    print("CLUSTER COORDINATE MAPPING TESTS")
    print("="*70)
    
    test_get_cluster_coordinates()
    test_cluster_density_map()
    test_3d_cluster_regions()
    test_visualize_cluster_coverage()
    test_practical_use_case()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
