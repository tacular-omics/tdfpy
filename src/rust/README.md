# Rust Extension for TDFpy

This directory contains a PyO3-based Rust extension for accelerating performance-critical operations in TDFpy.

## Overview

The Rust extension provides optimized implementations of computationally intensive functions:

- `merge_peaks`: Greedy clustering algorithm for peak centroiding with m/z and ion mobility tolerances

## Performance Benefits

The Rust implementation provides significant speedups over pure Python/NumPy:
- Reduced memory allocations through direct array manipulation
- Better CPU cache utilization with sorted data structures
- Native binary search and vectorized operations
- No Python interpreter overhead in tight loops

## Building the Extension

### Prerequisites

1. **Rust toolchain**: Install via [rustup](https://rustup.rs/)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Maturin**: Python package for building Rust extensions
   ```bash
   uv pip install maturin
   ```

### Development Build

Build and install the extension in development mode:

```bash
make install-rust
# or
uv run maturin develop --release
```

This compiles the Rust code with optimizations and installs it into your Python environment.

### Testing with Rust Extension

```bash
make test-rust
```

### Production Build

Build wheels for distribution:

```bash
make build-rust
# or
uv run maturin build --release
```

Wheels will be created in `target/wheels/`.
