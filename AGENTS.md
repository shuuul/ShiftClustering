# AGENTS.md

## Project Overview

ShiftClustering is a Python package providing fast clustering algorithms implemented in C++/Cython with OpenMP parallel support. It targets cryo-EM workflows and general-purpose clustering tasks.

**Algorithms:**
- **MeanShiftPP** — Optimized mean shift with grid-based binning (from [meanshiftpp](https://github.com/jenniferjang/meanshiftpp))
- **GridShift** — Grid-based iterative clustering (from [GridShift](https://github.com/abhisheka456/GridShift))
- **LocalShift** — 3D point cloud optimization for cryo-EM density maps (based on MAINMAST)

## Architecture

```
shiftclustering/
├── include/           # C++ headers (algorithm kernels)
│   ├── meanshiftpp.h  # shift_cy(): bin-and-shift iteration
│   ├── gridshift.h    # grid_cluster(): grid-based clustering with convergence
│   ├── localshift.h   # localshift_single_cy(): per-point Gaussian kernel shift
│   └── utils.h        # generate_offsets_cy(): neighbor offset generation
├── src/               # Cython wrappers (.pyx)
│   ├── _meanshiftpp.pyx
│   ├── _localshift.pyx
│   └── _gridshift.pyx
└── __init__.py        # Public API (class + functional interfaces)
```

Each algorithm follows the same pattern: a C++ header contains the core computation, a Cython `.pyx` file wraps it with Python type conversion and memory management, and `__init__.py` re-exports both a functional interface (`meanshiftpp()`, `gridshift()`, `localshift()`) and a scikit-learn-style class (`MeanShiftPP`, `GridShift`, `LocalShift`).

## Build System

- **Build backend:** scikit-build-core (CMake-based)
- **Cython compilation:** CMake invokes `cython --cplus` to generate C++ from `.pyx`, then compiles as Python extension modules
- **C++ standard:** C++17
- **Parallelism:** OpenMP (used in LocalShift's `prange`)
- **Package manager:** uv

### Building

```bash
uv sync                     # install + build
uv sync --all-extras        # include benchmark deps (numba, sklearn, torch)
uv sync --reinstall         # force rebuild of C++ extensions
```

### Key Build Files

- `pyproject.toml` — package metadata, dependencies, scikit-build-core config
- `CMakeLists.txt` — Cython → C++ → shared library pipeline
- `.python-version` — pinned Python version for uv

## Coding Conventions

- **Docstrings:** NumPy-style (`Parameters`, `Returns`, `Examples` sections)
- **Data types:** All arrays use `float32` / `int32` for C++ interop. Input arrays are coerced via `np.ascontiguousarray(..., dtype=np.float32)`.
- **Cython directives:** `boundscheck=False`, `wraparound=False`, `initializedcheck=False`, `language_level=3`
- **Memory:** Typed memoryviews (`float32_t[:, ::1]`) for zero-copy C++ access
- **GIL:** Released during C++ calls via `with nogil` / `prange`

## Common Tasks

### Adding a new algorithm

1. Write the C++ kernel in `shiftclustering/include/newalgo.h`
2. Create a Cython wrapper in `shiftclustering/src/_newalgo.pyx` with both a functional interface and a class interface
3. Export from `shiftclustering/__init__.py`
4. CMake auto-discovers `.pyx` files via `file(GLOB ...)` — no CMakeLists.txt changes needed

### Running benchmarks

```bash
uv sync --all-extras
python benchmark/benchmark_localshift_numba.py
python benchmark/benchmark_mspp_numba.py
python benchmark/benchmark_mspp_torch.py
python benchmark/benchmark_sklearn.py
```

## Dependencies

- **Runtime:** `numpy>=1.26`
- **Build:** `scikit-build-core>=0.11`, `cython>=3.1`, `numpy>=1.26`
- **Benchmark (optional):** `numba>=0.60`, `scikit-learn>=1.7`, `torch>=2.7.0`

## Testing

No test suite exists yet. When adding tests, use pytest and place them in a `tests/` directory. Test against small synthetic datasets to validate cluster label correctness and numerical stability.

## Important Notes

- The `__version__` in `__init__.py` and `version` in `pyproject.toml` must stay in sync.
- LocalShift uses OpenMP via `cython.parallel.prange`; MeanShiftPP and GridShift are single-threaded.
- All `.pyx` files define `NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION` to suppress NumPy C-API deprecation warnings.
- If you use LocalShift in research, cite: Terashi & Kihara, *Nature Communications* 9, 1618 (2018).
