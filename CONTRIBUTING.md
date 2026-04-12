# Contributing

## Build and test

From the repository root:

```bash
cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Release builds (recommended before benchmarking or profiling):

```bash
cmake -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Requirements: **C++17**, **CMake ≥ 3.14**, **OpenCV** (`core`, `imgproc`, `imgcodecs`), **GoogleTest** (system package or FetchContent).

## Formatting

The project uses **Google style** with project-specific tweaks (see `.clang-format`). Format changed sources before opening a pull request:

```bash
clang-format -i path/to/file.cpp
```

## Compiler warnings

`CMakeLists.txt` enables `-Wall`, `-Wextra`, `-Wpedantic`, `-Wshadow`, `-Wconversion`, `-Wsign-conversion`, and `-Werror`. New code must compile cleanly.

## Optional clang-tidy

Configure with `-DENABLE_CLANG_TIDY=ON` to run **clang-tidy** during compilation (requires `clang-tidy` on `PATH`).

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs **`clang-format --dry-run --Werror`** on tracked `*.cpp` / `*.h`, then builds **Debug** and **Release**, runs **`clang-tidy`** on every `src/*.cpp` (using the Debug compilation database), **ctest**, and a **smoke test** on `pyramid_demo` with `data/test_512.png`. Push to `main`/`master` or open a pull request against those branches to validate your changes.

## Tests and API boundaries

- **Public API** lives under `include/pyramid/`. Tests in `tests/laplacian_pyramid_test.cpp` should depend only on those headers, not on implementation details in `src/`.
- **Internal modules** (`image_arithmetic`, `separable_filter`) have dedicated tests in `tests/image_arithmetic_test.cpp` and `tests/separable_filter_test.cpp`.

When adding behavior, extend or add GoogleTest cases and keep the demo (`pyramid_demo`) in sync if user-visible behavior changes.
