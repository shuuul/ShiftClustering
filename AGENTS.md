# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-12
**Commit:** 6582df0
**Branch:** main

## OVERVIEW
Python/C++ hybrid package providing fast clustering algorithms (MeanShiftPP, GridShift, LocalShift) with nanobind bindings and OpenMP parallelism.

## STRUCTURE
```
./
├── shiftclustering/           # Main package
│   ├── include/              # C++ kernels (4 headers)
│   ├── src/                  # nanobind bindings (3 .cpp)
│   ├── _meanshiftpp.py       # Python wrapper
│   ├── _gridshift.py
│   ├── _localshift.py
│   └── __init__.py
├── tests/                    # pytest suite (192 lines)
├── benchmark/                # Performance comparisons
├── CMakeLists.txt
└── pyproject.toml
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new algorithm | `include/*.h` → `src/*_nb.cpp` → `_*.py` | Three-layer pattern |
| Fix bug | Python: `_*.py`, C++: `include/*.h` | Check wrapper first |
| Run tests | `tests/test_clustering.py` | `uv run pytest tests/ -v` |
| Run benchmarks | `benchmark/*.py` | Needs `--all-extras` |
| Build package | Root | `uv sync` |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `meanshiftpp()` | func | `_meanshiftpp.py:15` | Functional API |
| `MeanShiftPP` | class | `_meanshiftpp.py:70` | sklearn-style API |
| `gridshift()` | func | `_gridshift.py` | Grid-based clustering |
| `localshift()` | func | `_localshift.py` | 3D cryo-EM optimization |
| `_shift_cy()` | C++ | `src/_meanshiftpp_nb.cpp` | GIL-released kernel |
| `_generate_offsets()` | C++ | `src/*_nb.cpp` | Neighbor offset gen |

## CONVENTIONS
- **Docstrings:** NumPy-style (Parameters/Returns/Examples)
- **Array dtypes:** float32/int32 only; coerce via `np.ascontiguousarray(..., dtype=np.float32)`
- **nanobind:** Use `nb::ndarray<T, nb::numpy, nb::c_contig, nb::ndim<N>>` + `nb::gil_scoped_release`
- **API style:** Always provide both functional (`func(X, ...)`) AND class (`Class.fit_predict()`) interfaces
- **Version:** `__version__` in `__init__.py` MUST match `version` in `pyproject.toml`

## ANTI-PATTERNS (THIS PROJECT)
- NEVER use `as any`, `@ts-ignore`, `@ts-expect-error` (no TypeScript here, but similar discipline for types)
- NEVER commit without running tests: `uv run pytest tests/ -v`
- NEVER skip dtype coercion—C++ expects float32/int32

## UNIQUE STYLES
- **Three-layer architecture:** C++ header → nanobind binding → Python wrapper per algorithm
- **nanobind (not Cython):** Modern header-only binding library
- **OpenMP optional:** Only LocalShift uses it; falls back to sequential if unavailable
- **No CI pipeline:** Manual testing required (gap)

## COMMANDS
```bash
uv sync                      # install + build
uv sync --all-extras         # include benchmark + test deps
uv sync --reinstall          # force rebuild C++ extensions
uv run pytest tests/ -v       # run tests
```

## NOTES
- LocalShift is the ONLY parallel algorithm (OpenMP); MeanShiftPP/GridShift are single-threaded
- If using LocalShift in research, cite: Terashi & Kihara, *Nature Communications* 9, 1618 (2018)
- C++ sources nested inside Python package (`shiftclustering/src/`)—unconventional but works
