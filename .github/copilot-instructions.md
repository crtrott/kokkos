# Kokkos Development Guide for AI Coding Agents

## Repository Overview

**Kokkos** is a C++ performance portability programming ecosystem that provides abstractions for parallel execution and data management on HPC platforms. It supports multiple backend programming models including CUDA, HIP, SYCL, HPX, OpenMP, and C++ threads.

- **Language**: C++ (requires C++20 minimum, supports C++23/C++26)
- **Build System**: CMake (minimum version 3.16)
- **Repository Size**: ~100MB, primarily C++ source with CMake configuration
- **Project Type**: High-performance computing library with core, algorithms, containers, and SIMD components
- **License**: Apache-2.0 WITH LLVM-exception

## Project Structure

### Key Directories
- **`core/`**: Core Kokkos library implementation with execution spaces, memory spaces, and parallel patterns
  - `core/src/`: Source files for different backends (Serial, OpenMP, CUDA, HIP, SYCL, etc.)
  - `core/unit_test/`: Comprehensive unit tests organized by backend (serial/, cuda/, openmp/, etc.)
  - `core/perf_test/`: Performance tests
- **`algorithms/`**: Kokkos algorithms library (sorting, random numbers, etc.)
- **`containers/`**: Container implementations (DualView, UnorderedMap, etc.)
- **`simd/`**: SIMD abstraction library
- **`cmake/`**: CMake configuration scripts and modules
- **`scripts/`**: Utility scripts for formatting, testing, copyright checks
- **`bin/`**: Build tools including `nvcc_wrapper`, `kokkos_launch_compiler`
- **`example/`**: Usage examples and tutorials
- **`benchmarks/`**: Performance benchmarks

### Component Libraries
The build produces these libraries (all typically built as static libraries):
1. `kokkoscore` - Core library
2. `kokkoscontainers` - Container library
3. `kokkosalgorithms` - Algorithms library
4. `kokkossimd` - SIMD library

## Build Instructions

### Prerequisites
- **Compiler**: C++20-capable compiler (g++ 13.3+, clang++ 18.1+, Intel icpc/icpx, etc.)
- **CMake**: Version 3.22 or later
- **Python 3**: Required for testing infrastructure
- **Optional**: clang-format 16.0 for code formatting, cmake-format for CMake file formatting

### Standard Build Workflow

**ALWAYS follow this exact sequence for building:**

```bash
# 1. Create a separate build directory (NEVER build in-source)
mkdir build
cd build

# 2. Configure with CMake
cmake -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DKokkos_ENABLE_SERIAL=ON \
      ..

# 3. Build the library (use -j4 for CI environments to avoid memory issues)
cmake --build . --parallel 4

# 4. Optional: Install
cmake --build . --target install
```

**Build Timing**: Without tests, build takes ~10-15 seconds with `-j4`. With tests enabled, build takes 5+ minutes.

### Key CMake Configuration Options

**Backend Selection** (at least one must be enabled):
- `-DKokkos_ENABLE_SERIAL=ON` - Serial backend (CPU, default if no host backend specified)
- `-DKokkos_ENABLE_OPENMP=ON` - OpenMP backend (CPU parallel)
- `-DKokkos_ENABLE_THREADS=ON` - C++ threads backend (CPU parallel)
- `-DKokkos_ENABLE_CUDA=ON` - NVIDIA CUDA backend (GPU)
- `-DKokkos_ENABLE_HIP=ON` - AMD HIP backend (GPU)
- `-DKokkos_ENABLE_SYCL=ON` - Intel SYCL backend (GPU/CPU)
- `-DKokkos_ENABLE_HPX=ON` - HPX backend (experimental)

**Build Options**:
- `-DKokkos_ENABLE_TESTS=ON` - Build unit tests (default: OFF)
- `-DKokkos_ENABLE_BENCHMARKS=ON` - Build benchmarks (default: OFF)
- `-DKokkos_ENABLE_EXAMPLES=ON` - Build examples (default: OFF)
- `-DCMAKE_BUILD_TYPE=<type>` - Release, Debug, RelWithDebInfo (default: RelWithDebInfo if not specified)
- `-DBUILD_SHARED_LIBS=ON` - Build shared libraries instead of static (default: OFF)
- `-DKokkos_ENABLE_COMPILER_WARNINGS=ON` - Enable all compiler warnings
- `-DCMAKE_CXX_STANDARD=<17|20|23>` - C++ standard version (default: 17)

**Important**: If modifying GPU-related code, you MUST enable the appropriate backend (CUDA/HIP/SYCL).

## Testing

### Running Tests

**ALWAYS configure with tests enabled to run tests:**

```bash
# Configure with tests
cmake -DCMAKE_CXX_COMPILER=g++ \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_TESTS=ON \
      ..

# Build all tests
cmake --build . --parallel 4

# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R <test_name> --output-on-failure

# Run tests in parallel (use with caution in CI)
ctest -j2 --output-on-failure
```

**Test Timing**: Full test suite can take 10+ minutes to build and several minutes to run depending on backend and parallelism.

### Test Organization
- Tests are located in `<component>/unit_test/` directories
- Tests are organized by backend in subdirectories (e.g., `serial/`, `cuda/`, `openmp/`)
- Each backend has its own test executables
- Use `ctest --timeout 2000` for long-running tests

## Code Formatting and Style

### C++ Code Formatting

**REQUIRED**: All C++ code MUST be formatted with clang-format version 16.0 exactly.

```bash
# Check if you have the correct version
clang-format --version  # Must show version 16.0.x

# Format all tracked files (from repository root)
./scripts/apply-clang-format

# Or set environment variable to use specific clang-format
CLANG_FORMAT_EXE=/path/to/clang-format-16 ./scripts/apply-clang-format
```

**Version requirement**: The `apply-clang-format` script will refuse to run with any version other than 16.0. Different clang-format versions can produce different formatting, causing CI failures. If you don't have version 16.0, either install it or use the `CLANG_FORMAT_EXE` environment variable to point to the correct binary.

**Formatting configuration**: `.clang-format` at repository root (based on Google style with modifications)

### CMake Formatting

CMake files should be formatted using cmake-format:

```bash
# Format CMake files (configuration in .cmake-format.py)
cmake-format --config-files .cmake-format.py --in-place <file.cmake>
```

### Copyright Headers

All source files must have the correct copyright header. Check with:
```bash
./scripts/check-copyright
```

## Continuous Integration

### GitHub Actions Workflows

The repository uses multiple CI workflows (`.github/workflows/`):

1. **`continuous-integration-linux.yml`** - Primary Linux testing (g++, clang++, Intel compilers)
2. **`continuous-integration-windows.yml`** - Windows testing
3. **`continuous-integration-osx.yml`** - macOS testing  
4. **`clang-format-check.yml`** - Enforces clang-format version 16.0
5. **`cmake-format-check.yml`** - Enforces CMake formatting
6. **`codeql.yml`** - Security analysis

### CI Build Configuration

CI builds use these settings (replicate locally for CI consistency):
```bash
cmake -B builddir \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DBUILD_SHARED_LIBS=ON \
  -DKokkos_ENABLE_HWLOC=ON \
  -DKokkos_ENABLE_<BACKEND>=ON \
  -DKokkos_ENABLE_TESTS=ON \
  -DKokkos_ENABLE_BENCHMARKS=ON \
  -DKokkos_ENABLE_EXAMPLES=ON \
  -DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
  -DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
  -DCMAKE_CXX_FLAGS="-Werror" \
  -DCMAKE_CXX_COMPILER=<compiler> \
  -DCMAKE_BUILD_TYPE=<Release|Debug>

cmake --build builddir --parallel 4
cd builddir && ctest --output-on-failure
```

**Important CI Notes**:
- CI uses `--parallel 4` for builds to avoid memory issues
- Tests run with `ctest --output-on-failure` 
- Some workflows use `ctest --timeout 2000` for long tests
- Clang-format check will FAIL if not using version 16.0 exactly
- CI tests both Debug and Release builds

### Pre-commit Validation Checklist

Before committing changes, ALWAYS:
1. ✅ Run `./scripts/apply-clang-format` (requires clang-format 16.0)
2. ✅ Format any modified CMake files with cmake-format
3. ✅ Run `./scripts/check-copyright` to verify headers
4. ✅ Build with warnings enabled: `-DKokkos_ENABLE_COMPILER_WARNINGS=ON -DCMAKE_CXX_FLAGS="-Werror"`
5. ✅ Run relevant tests with `ctest`
6. ✅ If modifying core functionality, test with multiple backends if possible

## Common Build Issues and Workarounds

### Issue: In-source build error
**Error**: "FATAL: In-source builds are not allowed"
**Solution**: Always create a separate build directory:
```bash
mkdir build && cd build && cmake ..
```

### Issue: Long build times with tests
**Problem**: Building with tests takes 5+ minutes
**Solution**: 
- For iterative development, build without tests initially: `-DKokkos_ENABLE_TESTS=OFF`
- Only enable tests when validating changes
- Use `-j4` instead of `-j$(nproc)` in CI environments

### Issue: Wrong clang-format version
**Error**: "This indent script requires clang-format version 16.0"
**Solution**: 
```bash
# Install clang-format 16 or use environment variable
CLANG_FORMAT_EXE=/usr/bin/clang-format-16 ./scripts/apply-clang-format
```

### Issue: Cray compiler linking
**Problem**: Static linking breaks on Cray systems
**Solution**: Set `CRAYPE_LINK_TYPE=dynamic` environment variable

### Issue: Fortran linker in mixed code
**Problem**: CMake uses Fortran linker instead of C++ linker
**Solution**: Requires CMake 3.18+, or ensure C++ is the link language

## Making Code Changes

### Workflow for Code Changes

1. **Explore relevant code**:
   - Core functionality: `core/src/`
   - Look for existing similar implementations
   - Check corresponding tests in `core/unit_test/`

2. **Make minimal changes**:
   - Modify only necessary files
   - Follow existing code patterns
   - Maintain consistency with surrounding code

3. **Format code immediately**:
   ```bash
   ./scripts/apply-clang-format
   ```

4. **Build and test iteratively**:
   ```bash
   # Quick build check (no tests)
   cd build && cmake --build . --parallel 4
   
   # Full test validation
   cd build_test && cmake --build . --parallel 4 && ctest --output-on-failure
   ```

5. **Verify no warnings**:
   ```bash
   cmake -DKokkos_ENABLE_COMPILER_WARNINGS=ON -DCMAKE_CXX_FLAGS="-Werror" ..
   cmake --build . --parallel 4
   ```

### Adding New Features

When adding new CMake options, TPLs, or compiler flags, see `cmake/README.md` for detailed guidelines on:
- Using `KOKKOS_OPTION` and `KOKKOS_ENABLE_OPTION`
- Adding config macros to `KokkosCore_config.h.in`
- Importing and exporting third-party libraries
- Following the modern CMake philosophy

## Quick Reference

### Essential Files
- `CMakeLists.txt` - Main CMake configuration
- `BUILD.md` - Detailed build instructions for users
- `CONTRIBUTING.md` - Contribution guidelines
- `.clang-format` - C++ formatting rules (clang-format 16.0)
- `.cmake-format.py` - CMake formatting rules
- `cmake/kokkos_enable_devices.cmake` - Backend device options
- `cmake/kokkos_enable_options.cmake` - Build options
- `cmake/README.md` - CMake development guide

### Essential Commands
```bash
# Format code
./scripts/apply-clang-format

# Check copyright
./scripts/check-copyright

# Minimal build (fastest)
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_SERIAL=ON ..
cmake --build . --parallel 4

# Build with tests
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=ON ..
cmake --build . --parallel 4
ctest --output-on-failure

# Validate like CI
cmake -DCMAKE_CXX_COMPILER=g++ \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_TESTS=ON \
      -DKokkos_ENABLE_COMPILER_WARNINGS=ON \
      -DCMAKE_CXX_FLAGS="-Werror" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --parallel 4
ctest --output-on-failure
```

## Important Notes

- **NEVER** build in the source directory - always use a separate build directory
- **ALWAYS** use clang-format version 16.0 exactly - other versions will fail CI
- **ALWAYS** use `-j4` for parallel builds in CI to avoid memory issues
- **Use these instructions as your primary reference** - they are validated and comprehensive. Search for additional information if something is unclear, appears incorrect, or if you encounter an error not documented here.

When in doubt, refer to `BUILD.md` for user-facing build documentation or `cmake/README.md` for build system development details.
