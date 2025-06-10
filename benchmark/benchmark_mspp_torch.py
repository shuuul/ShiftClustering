"""
PyTorch vs C++/Cython MeanShiftPP Benchmark

This benchmark compares the performance and accuracy of:
- PyTorch implementation of MeanShift++ (CPU only)
- C++/Cython implementation from shiftclustering.MeanShiftPP

Both implementations should produce similar results with proper parameter mapping:
- bandwidth: same in both
- n_steps (torch) = max_iter (MeanShiftPP) 
- tol (torch) = threshold (MeanShiftPP)
- base: 3 in both (corresponding to (-1, 0, 1) offsets)
"""

import statistics
import time

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

from shiftclustering import MeanShiftPP


def meanshiftpp_torch(X: torch.Tensor, bandwidth: float, n_steps: int, tol: float = 1e-3,
                        base: int = 3) -> torch.Tensor:
    """
    Mean shift clustering with PyTorch batch operations.

    Parameters
    ----------
    X : torch.Tensor
        (n, d) array of points.
    bandwidth : float
        Radius for binning points.
    n_steps : int
        Number of iterations.
    tol : float
        Tolerance for convergence.
    base : int
        Base for offsets. Default is 3.

    Returns
    -------
    X_shifted : torch.Tensor
        (n, d) array of new points after mean shift clustering.
    """

    n, d = X.shape

    # Generate offsets
    ranges = [torch.arange(-1, base - 1) for _ in range(d)]
    mesh = torch.meshgrid(ranges, indexing='ij')
    offsets = torch.stack(mesh, dim=-1).reshape(-1, d)
    offsets = offsets.to(X.device)

    X_shifted = X.clone().detach()

    for i in range(n_steps):
        bins = (X_shifted / bandwidth).int()  # Shape: [n, d]

        # Create a large tensor for all shifted bins
        all_shifted_bins = bins.unsqueeze(1) + offsets
        all_shifted_bins_flat = all_shifted_bins.reshape(-1, d)

        # Unique bins and their inverse indices
        unique_bins, inverse_indices = torch.unique(all_shifted_bins_flat, dim=0, return_inverse=True)

        # Compute sum and count for each unique bin
        sum_per_bin = torch.zeros_like(unique_bins, dtype=torch.float, device=X.device)
        sum_per_bin = sum_per_bin.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, d),
                                               X_shifted.repeat(1, base ** d).reshape(-1, d))

        count_per_bin = torch.zeros(len(unique_bins), dtype=torch.float, device=X.device)
        count_per_bin = count_per_bin.scatter_add_(0, inverse_indices,
                                                   torch.ones(n * base ** d, dtype=torch.float, device=X.device))

        # Map sums and counts back to the original bins
        sum_mapped = sum_per_bin[inverse_indices]  # Shape: [n * base ** d, d]
        count_mapped = count_per_bin[inverse_indices]  # Shape: [n * base ** d]

        # Compute new positions
        X_shifted_new = (sum_mapped.reshape(n, -1, d).sum(dim=1) / count_mapped.reshape(n, -1).sum(dim=1).unsqueeze(1))

        # Check for convergence
        if torch.max(torch.norm(X_shifted_new - X_shifted, dim=1)) <= tol:
            print(f"PyTorch MeanShift++ converged at {i + 1} steps.")
            break

        X_shifted = X_shifted_new.clone()

    return X_shifted


def torch_meanshift_clustering(X: torch.Tensor, bandwidth: float, n_steps: int, tol: float = 1e-3) -> np.ndarray:
    """
    Complete clustering pipeline using PyTorch MeanShift++.
    
    Returns cluster labels similar to MeanShiftPP.fit_predict().
    """
    X_shifted = meanshiftpp_torch(X, bandwidth, n_steps, tol)
    
    # Convert to numpy for clustering
    X_shifted_np = X_shifted.cpu().numpy()
    
    # Find unique cluster centers (similar to MeanShiftPP)
    cluster_centers, labels = np.unique(X_shifted_np, return_inverse=True, axis=0)
    
    return labels


def benchmark_implementations(
    X_np: np.ndarray, 
    bandwidth: float, 
    max_iter: int = 300, 
    threshold: float = 1e-4,
    n_runs: int = 3
) -> dict[str, dict]:
    """
    Benchmark both implementations with identical parameters.
    
    Parameters
    ----------
    X_np : np.ndarray
        Input data as numpy array
    bandwidth : float
        Bandwidth parameter
    max_iter : int
        Maximum iterations (n_steps for torch, max_iter for MeanShiftPP)
    threshold : float
        Convergence threshold (tol for torch, threshold for MeanShiftPP)
    n_runs : int
        Number of runs for statistical stability
        
    Returns
    -------
    Dict with results for both implementations
    """
    
    results = {
        'torch': {'times': [], 'labels': [], 'n_clusters': []},
        'cython': {'times': [], 'labels': [], 'n_clusters': []}
    }
    
    # Convert to torch tensor (CPU only)
    X_torch = torch.from_numpy(X_np.astype(np.float32))
    
    print(f"Benchmarking with {len(X_np)} samples, {X_np.shape[1]} features")
    print(f"Parameters: bandwidth={bandwidth}, max_iter={max_iter}, threshold={threshold}")
    
    # Benchmark PyTorch implementation
    print("\nTesting PyTorch implementation...")
    for run in range(n_runs):
        start_time = time.time()
        labels_torch = torch_meanshift_clustering(X_torch, bandwidth, max_iter, threshold)
        end_time = time.time()
        
        elapsed = end_time - start_time
        n_clusters = len(np.unique(labels_torch))
        
        results['torch']['times'].append(elapsed)
        results['torch']['labels'].append(labels_torch)
        results['torch']['n_clusters'].append(n_clusters)
        
        print(f"  Run {run+1}: {elapsed:.4f}s, {n_clusters} clusters")
    
    # Benchmark C++/Cython implementation
    print("\nTesting C++/Cython implementation...")
    for run in range(n_runs):
        # Create fresh instance for each run
        ms = MeanShiftPP(bandwidth=bandwidth, threshold=threshold, max_iter=max_iter)
        
        start_time = time.time()
        labels_cython = ms.fit_predict(X_np)
        end_time = time.time()
        
        elapsed = end_time - start_time
        n_clusters = len(np.unique(labels_cython))
        
        results['cython']['times'].append(elapsed)
        results['cython']['labels'].append(labels_cython)
        results['cython']['n_clusters'].append(n_clusters)
        
        print(f"  Run {run+1}: {elapsed:.4f}s, {n_clusters} clusters")
    
    return results


def calculate_metrics(results: dict[str, dict], X: np.ndarray, y_true: np.ndarray | None= None) -> dict[str, dict]:
    """Calculate performance and quality metrics."""
    metrics = {}
    
    for impl_name, data in results.items():
        times = data['times']
        labels_list = data['labels']
        n_clusters_list = data['n_clusters']
        
        # Use first run for quality metrics
        labels = labels_list[0]
        
        # Performance metrics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        mean_clusters = statistics.mean(n_clusters_list)
        
        # Quality metrics
        quality_metrics = {}
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
            try:
                quality_metrics['silhouette'] = silhouette_score(X, labels)
            except ValueError as e:
                print(f"Silhouette score not available: {e}")
                quality_metrics['silhouette'] = -1.0
        else:
            quality_metrics['silhouette'] = -1.0
            
        if y_true is not None:
            quality_metrics['ari'] = adjusted_rand_score(y_true, labels)
        
        metrics[impl_name] = {
            'mean_time': mean_time,
            'std_time': std_time,
            'mean_clusters': mean_clusters,
            'all_times': times,
            **quality_metrics
        }
    
    return metrics


def compare_results(results: dict[str, dict]) -> dict[str, float]:
    """Compare the similarity of clustering results between implementations."""
    torch_labels = results['torch']['labels'][0]
    cython_labels = results['cython']['labels'][0]
    
    # Calculate Adjusted Rand Index between the two implementations
    ari = adjusted_rand_score(torch_labels, cython_labels)
    
    return {'ari_between_implementations': ari}


def print_results(metrics: dict[str, dict], comparison: dict[str, float]) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'='*90}")
    print("PYTORCH vs C++/CYTHON MEANSHIFT++ BENCHMARK RESULTS")
    print(f"{'='*90}")
    
    print(f"{'Implementation':<15} {'Time (s)':<12} {'±Std':<8} {'Clusters':<10} {'Silhouette':<12} {'ARI':<8}")
    print("-" * 90)
    
    for impl_name, data in metrics.items():
        display_name = "PyTorch" if impl_name == "torch" else "C++/Cython"
        ari_str = f"{data.get('ari', 0.0):.4f}" if 'ari' in data else "N/A"
        
        print(f"{display_name:<15} "
              f"{data['mean_time']:<12.4f} "
              f"±{data['std_time']:<7.4f} "
              f"{data['mean_clusters']:<10.1f} "
              f"{data['silhouette']:<12.4f} "
              f"{ari_str:<8}")
    
    # Speedup analysis
    torch_time = metrics['torch']['mean_time']
    cython_time = metrics['cython']['mean_time']
    
    if torch_time < cython_time:
        speedup = cython_time / torch_time
        faster = "PyTorch"
    else:
        speedup = torch_time / cython_time
        faster = "C++/Cython"
    
    print(f"\nPerformance Analysis:")
    print(f"- {faster} is {speedup:.2f}x faster")
    print(f"- Implementation Agreement (ARI): {comparison['ari_between_implementations']:.4f}")
    
    if comparison['ari_between_implementations'] < 0.8:
        print("  ⚠️  Low agreement between implementations - results may differ significantly")
    elif comparison['ari_between_implementations'] > 0.95:
        print("  ✅ High agreement between implementations - results are very similar")
    else:
        print("  ⚠️  Moderate agreement between implementations")


def main():
    """Run the benchmark with different dataset configurations."""
    
    # Test configurations
    configs = [
        {"n_samples": 1000, "n_features": 2, "n_centers": 5, "bandwidth": 1.0},
        {"n_samples": 3000, "n_features": 3, "n_centers": 10, "bandwidth": 1.5},
        {"n_samples": 5000, "n_features": 4, "n_centers": 15, "bandwidth": 2.0},
    ]
    
    max_iter = 300
    threshold = 1e-4
    n_runs = 3
    random_state = 42
    
    print("PyTorch vs C++/Cython MeanShift++ Benchmark")
    print(f"PyTorch device: CPU only")
    print(f"Parameters: max_iter={max_iter}, threshold={threshold}, runs={n_runs}")
    print(f"Random state: {random_state}")
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*100}")
        print(f"TEST {i}/{len(configs)}: {config['n_samples']} samples, {config['n_features']}D, {config['n_centers']} centers")
        print(f"{'='*100}")
        
        # Generate dataset
        X, y_true = make_blobs(
            n_samples=config['n_samples'],
            n_features=config['n_features'], 
            centers=config['n_centers'],
            random_state=random_state,
            cluster_std=1.0
        )
        
        print(f"Dataset shape: {X.shape}")
        print(f"True clusters: {len(np.unique(y_true))}")
        
        # Run benchmark
        results = benchmark_implementations(
            X, config['bandwidth'], max_iter, threshold, n_runs
        )
        
        # Calculate metrics
        metrics = calculate_metrics(results, X, y_true)
        comparison = compare_results(results)
        
        # Print results
        print_results(metrics, comparison)
    
    print(f"\n{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print("Key Findings:")
    print("- Both implementations should produce similar clustering results")
    print("- C++/Cython typically faster due to optimized memory management")
    print("- PyTorch more flexible for GPU acceleration (not tested here)")
    print("- Parameter equivalence: n_steps=max_iter, tol=threshold, base=3")


if __name__ == "__main__":
    main()
