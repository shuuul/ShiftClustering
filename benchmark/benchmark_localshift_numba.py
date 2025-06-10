#!/usr/bin/env python3
"""
Benchmark script comparing Cython and Numba implementations of local shift algorithm.

This script creates a synthetic cryo-EM-like scenario with:
- A 3D reference density map (600x600x600)
- 1000 randomly positioned atoms
- Reference map populated with Gaussian density around atoms with radius sqrt(2)
- Performance comparison between Cython and Numba implementations

USAGE:
------
1. To run with just Numba (no build required):
   python benchmark/benchmark_localshift_numba.py

2. To run with both Cython and Numba:
   # First build the Cython extension:
   pip install -e .
   # Then run the benchmark:
   python benchmark/benchmark_localshift_numba.py

REQUIREMENTS:
-------------
- numpy
- numba
- shiftclustering (for Cython implementation)

The script benchmarks both Numba and Cython implementations.
"""

import time
import numpy as np
from numba import njit
import sys
import os

# Add the parent directory to sys.path to import shiftclustering
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shiftclustering import LocalShift, localshift


@njit
def localshift_numba(point_cd, reference, fmaxd, fsiv, n_steps, tol):
    """
    Numba implementation of local shift algorithm.
    
    Parameters
    ----------
    point_cd : ndarray, shape (n_points, 3)
        3D coordinates of points to be shifted (modified in-place)
    reference : ndarray, shape (D, H, W)
        3D reference density map
    fmaxd : float
        Maximum distance for neighbor search
    fsiv : float
        Kernel parameter for Gaussian weighting
    n_steps : int
        Maximum number of iterations per point
    tol : float
        Tolerance for convergence (squared distance)
        
    Returns
    -------
    ndarray
        Optimized 3D coordinates after local shift
    """
    cnt = point_cd.shape[0]
    ref_shape = reference.shape
    fsiv_neg = -1.5 * fsiv

    for i in range(cnt):
        for step in range(n_steps):
            pos = point_cd[i]
            stp = np.array([max(int(pos[0] - fmaxd), 0),
                            max(int(pos[1] - fmaxd), 0),
                            max(int(pos[2] - fmaxd), 0)], dtype=np.int32)
            endp = np.array([min(int(pos[0] + fmaxd + 1), ref_shape[0] - 1),
                             min(int(pos[1] + fmaxd + 1), ref_shape[1] - 1),
                             min(int(pos[2] + fmaxd + 1), ref_shape[2] - 1)], dtype=np.int32)

            pos2 = np.zeros(3, dtype=np.float32)
            dtotal = 0.0

            for xp in range(stp[0], endp[0]):
                for yp in range(stp[1], endp[1]):
                    for zp in range(stp[2], endp[2]):
                        offset = np.array([xp, yp, zp], dtype=np.float32)
                        d2 = (offset[0] - pos[0]) ** 2 + (offset[1] - pos[1]) ** 2 + (offset[2] - pos[2]) ** 2
                        kernel_weight = np.exp(fsiv_neg * d2) * reference[xp, yp, zp]
                        if kernel_weight > 0:
                            dtotal += kernel_weight
                            pos2[0] += kernel_weight * offset[0]
                            pos2[1] += kernel_weight * offset[1]
                            pos2[2] += kernel_weight * offset[2]

            if dtotal > 0:
                pos2 /= dtotal
                shift_dis_square = (pos[0] - pos2[0]) ** 2 + (pos[1] - pos2[1]) ** 2 + (pos[2] - pos2[2]) ** 2
                if shift_dis_square < tol:
                    break
                point_cd[i, 0] = pos2[0]
                point_cd[i, 1] = pos2[1]
                point_cd[i, 2] = pos2[2]

    return point_cd


def create_synthetic_data(map_shape=(600, 600, 600), n_atoms=1000, radius=np.sqrt(2)):
    """
    Create synthetic cryo-EM-like data for benchmarking.
    
    Parameters
    ----------
    map_shape : tuple
        Shape of the 3D reference map
    n_atoms : int
        Number of atoms to place randomly
    radius : float
        Radius for Gaussian density around atoms
        
    Returns
    -------
    tuple
        (atom_positions, reference_map)
    """
    print(f"Creating synthetic data: map_shape={map_shape}, n_atoms={n_atoms}, radius={radius:.3f}")
    
    # Generate random atom positions within the map bounds
    atom_positions = np.random.rand(n_atoms, 3) * np.array(map_shape)
    atom_positions = atom_positions.astype(np.float32)
    
    # Create empty reference map
    reference_map = np.zeros(map_shape, dtype=np.float32)
    
    # Populate reference map with Gaussian density around each atom
    print("Populating reference map with Gaussian densities...")
    sigma = radius / 2.0  # Convert radius to sigma for Gaussian
    
    for i, pos in enumerate(atom_positions):
        if i % 100 == 0:
            print(f"  Processing atom {i+1}/{n_atoms}")
            
        # Define region around atom
        x, y, z = pos
        x_min = max(0, int(x - 3*radius))
        x_max = min(map_shape[0], int(x + 3*radius + 1))
        y_min = max(0, int(y - 3*radius))
        y_max = min(map_shape[1], int(y + 3*radius + 1))
        z_min = max(0, int(z - 3*radius))
        z_max = min(map_shape[2], int(z + 3*radius + 1))
        
        # Create coordinate grids
        xx, yy, zz = np.meshgrid(
            np.arange(x_min, x_max),
            np.arange(y_min, y_max),
            np.arange(z_min, z_max),
            indexing='ij'
        )
        
        # Calculate distances and Gaussian weights
        dist_sq = (xx - x)**2 + (yy - y)**2 + (zz - z)**2
        gaussian = np.exp(-dist_sq / (2 * sigma**2))
        
        # Add to reference map (with some intensity variation)
        intensity = 0.5 + 0.5 * np.random.rand()  # Random intensity between 0.5 and 1.0
        reference_map[x_min:x_max, y_min:y_max, z_min:z_max] += intensity * gaussian
    
    print(f"Reference map statistics: min={reference_map.min():.3f}, max={reference_map.max():.3f}, mean={reference_map.mean():.3f}")
    
    # Add some noise to atom positions for the optimization
    noise = np.random.normal(0, 1.0, atom_positions.shape).astype(np.float32)
    noisy_positions = atom_positions + noise
    
    # Ensure positions are within bounds
    noisy_positions = np.clip(noisy_positions, 0, np.array(map_shape) - 1)
    
    return noisy_positions, reference_map


def benchmark_implementation(name, func, point_cd, reference, params, n_runs=3):
    """
    Benchmark a single implementation.
    
    Parameters
    ----------
    name : str
        Name of the implementation
    func : callable
        Function to benchmark
    point_cd : ndarray
        Initial atom positions
    reference : ndarray
        Reference density map
    params : dict
        Parameters for the algorithm
    n_runs : int
        Number of runs for averaging
        
    Returns
    -------
    tuple
        (average_time, result_positions)
    """
    print(f"\nBenchmarking {name}...")
    
    times = []
    results = []
    
    for run in range(n_runs):
        # Make a copy for each run
        points_copy = point_cd.copy()
        
        start_time = time.time()
        
        try:
            if name == "Cython":
                # Use class-based interface
                ls = LocalShift(**params)
                result = ls.fit_predict(points_copy, reference)
            elif name == "Cython (functional)":
                # Use functional interface
                result = func(points_copy, reference, **params)
            else:
                # Numba
                result = func(points_copy, reference, **params)
            
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            results.append(result.copy())
            
            print(f"  Run {run+1}: {elapsed:.3f}s")
            
        except Exception as e:
            print(f"  Run {run+1}: FAILED - {e}")
            return None, None
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
    
    return avg_time, results[0]  # Return average time and first result


def compare_results(results_dict, original_positions):
    """
    Compare results from different implementations.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping implementation names to result positions
    original_positions : ndarray
        Original noisy positions before optimization
    """
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    # Calculate movement statistics
    for name, positions in results_dict.items():
        if positions is not None:
            movement = np.linalg.norm(positions - original_positions, axis=1)
            print(f"\n{name}:")
            print(f"  Average movement: {movement.mean():.3f} ± {movement.std():.3f}")
            print(f"  Max movement: {movement.max():.3f}")
            print(f"  Min movement: {movement.min():.3f}")
    
    # Compare between implementations
    implementations = [name for name, pos in results_dict.items() if pos is not None]
    
    if len(implementations) >= 2:
        print(f"\nDifferences between implementations:")
        for i in range(len(implementations)):
            for j in range(i+1, len(implementations)):
                name1, name2 = implementations[i], implementations[j]
                pos1, pos2 = results_dict[name1], results_dict[name2]
                
                diff = np.linalg.norm(pos1 - pos2, axis=1)
                print(f"  {name1} vs {name2}:")
                print(f"    Average difference: {diff.mean():.6f} ± {diff.std():.6f}")
                print(f"    Max difference: {diff.max():.6f}")





def main():
    """Main benchmark function."""
    print("="*60)
    print("LOCAL SHIFT ALGORITHM BENCHMARK")
    print("Comparing Cython vs Numba implementations")
    print("="*60)
    
    # Parameters - you can adjust these for testing
    # Full size: (600, 600, 600) with 1000 atoms
    # Reduced size for testing: (200, 200, 200) with 100 atoms
    map_shape = (600, 600, 600)
    n_atoms = 1000
    radius = np.sqrt(2)
    
    # Algorithm parameters
    params = {
        'fmaxd': 5.0,
        'fsiv': 0.1,
        'n_steps': 50,
        'tol': 1e-6
    }
    
    print(f"Algorithm parameters: {params}")
    
    # Create synthetic data
    print("\n" + "-"*40)
    print("DATA GENERATION")
    print("-"*40)
    
    original_positions, reference_map = create_synthetic_data(map_shape, n_atoms, radius)
    
    print(f"Created {n_atoms} atoms in {map_shape} reference map")
    print(f"Reference map non-zero elements: {np.count_nonzero(reference_map)}")
    
    # Benchmark implementations
    print("\n" + "-"*40)
    print("BENCHMARKING")
    print("-"*40)
    
    results = {}
    times = {}
    
    # Numba implementation
    print("Warming up Numba...")
    # Warm-up run for Numba JIT compilation
    test_points = original_positions[:10].copy()
    test_ref = reference_map[100:150, 100:150, 100:150].copy()
    localshift_numba(test_points, test_ref, params['fmaxd'], params['fsiv'], 2, params['tol'])
    
    # Full benchmark
    time_numba, result_numba = benchmark_implementation(
        "Numba", localshift_numba, original_positions, reference_map, params
    )
    times['Numba'] = time_numba
    results['Numba'] = result_numba
    
    # Cython implementations
    # Class-based interface
    time_cython, result_cython = benchmark_implementation(
        "Cython", None, original_positions, reference_map, params
    )
    times['Cython'] = time_cython
    results['Cython'] = result_cython
    
    # Functional interface
    time_cython_func, result_cython_func = benchmark_implementation(
        "Cython (functional)", localshift, original_positions, reference_map, params
    )
    times['Cython (functional)'] = time_cython_func
    results['Cython (functional)'] = result_cython_func
    
    # Compare results
    compare_results(results, original_positions)
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    valid_times = {name: time_val for name, time_val in times.items() if time_val is not None}
    
    if valid_times:
        fastest_name = min(valid_times.keys(), key=lambda k: valid_times[k])
        fastest_time = valid_times[fastest_name]
        
        print(f"Fastest implementation: {fastest_name} ({fastest_time:.3f}s)")
        print("\nSpeedup factors (relative to slowest):")
        
        slowest_time = max(valid_times.values())
        for name, time_val in sorted(valid_times.items(), key=lambda x: x[1]):
            speedup = slowest_time / time_val
            print(f"  {name}: {speedup:.2f}x")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
