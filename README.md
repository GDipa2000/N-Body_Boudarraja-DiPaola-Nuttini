
Elena Nuttini
# NBody-Boudarraja-DiPaola-Nuttini
Implementation of "N-Body Problem" for AMSC course.

## Overview
This program implements two solvers for the N-Body Problem simulation for elastic collisions between particles subject to the same force (gravitational or Coulomb): the first is the standard direct-sum algorithm, which takes advantage of the symmetry of the pairwise computation (Newton's third law) and Euler's method to update velocities and positions. The second solver is based on the Barnes-Hut algorithm, which speeds up the computation using a quadtree to group distant particles by their center of mass instead of evaluating every pairwise interaction; this solver uses Velocity Verlet integration for a more accurate estimate of the new positions and velocities.

The simulation can run in 2D or 3D for the standard solver, and in 2D only for Barnes-Hut, using the same executable and command-line flags; the visualizer displays the result accordingly. Each solver is available both in a serial and an OpenMP-parallel version, for four simulation modes in total.

## Prerequisites

**To build and run the simulation:**
- A C++17 compiler with OpenMP support (`g++` is assumed throughout this README)
- CMake ≥ 3.10, only if you use the "computational part only" workflow described below

**To run the visual simulation (`animation.py`):**
- Python 3
- `matplotlib` (with a working Tk backend — `animation.py` explicitly selects `TkAgg`, so a headless machine without a display will not be able to show the animation window)
- `numpy`

Nothing needs to be installed manually for the C++ side beyond a working `g++`/OpenMP toolchain: both workflows below compile the project themselves.

## How to run the visual simulation

The visual simulation is launched by the Python script that displays the resulting animation. Run it from `NBody AMSC/source/graphics`:

```bash
cd "NBody AMSC/source/graphics"
python3 animation.py
```

`animation.py` compiles the computational program itself before running it, with:

```bash
g++ -fopenmp -I../utils ../main/main.cpp -o main
```

i.e. it expects the header files (`particle.hpp`, `force.hpp`, `quadtreeNode.hpp`, etc.) in `NBody AMSC/source/utils`, and `main.cpp` in `NBody AMSC/source/main`; the resulting `main` executable is created directly inside `source/graphics`, so the default relative output paths in `main.cpp` resolve correctly without any extra setup.

**Running with no arguments** (`python3 animation.py`) runs the freshly compiled `main` with no flags. Since `main.cpp` requires at least one argument to skip its interactive prompt, this makes the program ask:

```
Enter 'd' to run the default simulation:
```

Press `d` and Enter to run the default 2D, parallel, gravitational-force simulation. Pressing anything else only prints the flag list and exits *without producing any output file* — if that happens, `animation.py` will fail immediately afterwards with a `FileNotFoundError` on `Info.txt`, since there is nothing to visualize.

**To pass parameters**, provide them as a single quoted string, exactly as they would appear on the command line; `animation.py` splits the string on spaces and forwards each piece as a separate argument, bypassing the interactive prompt entirely:

```bash
python3 animation.py "-dim 2 -simT 0"
```

### Two flags that do not work with `animation.py`

- **`-file <path>`** — changes where the C++ program *writes* `Info.txt`, but `animation.py` always *reads* a file literally named `Info.txt` in the current directory, regardless of this flag. Do not pass `-file` when using `animation.py`; it is only meant for the CMake-built executable described below, where you manage the output location yourself.
- **`NBODY_NO_IO=1`** (see the environment variable below) disables all file output. Since `animation.py` depends entirely on the coordinate files being written, never set this variable when running through `animation.py` — only use it for pure timing runs of the CMake-built executable.

### A known limitation: `-simT 2` cannot currently be visualized

`main.cpp` decides how many `Coordinates_i.txt` files to expect based on whether the simulation is serial (`simType == 0`) or not — but it tests `simType` against zero specifically, rather than against "is this mode serial". Since `-simT 2` (serial Barnes-Hut) is a serial mode but is *not* zero, `Info.txt` ends up declaring as many coordinate files as there are available threads, while `serialBarnesHut` always writes to a single `Coordinates_0.txt`. If you run:

```bash
python3 animation.py "-simT 2"
```

on a machine with more than one available thread, `animation.py` will try to open `Coordinates_1.txt` and fail with a `FileNotFoundError`, because that file is never created. This does not affect the correctness of the simulation itself — only its visualization. Serial brute-force (`-simT 0`), parallel brute-force (`-simT 1`), and parallel Barnes-Hut (`-simT 3`) are unaffected and visualize normally.

## How to run ONLY the computational part

The computational program is defined by `NBody AMSC/source/main/main.cpp`. From the repository root, configure and compile it with CMake:

```bash
cd "NBody AMSC"
cmake -S . -B build
cmake --build build
```

The executable is created at `build/nbody`. CMake detects OpenMP and enables it automatically, so `-fopenmp` and the include path do not need to be entered manually.

To execute the CMake-built program, run it from `source/main` so that the relative output paths used by the program point to `source/graphics`:

```bash
cd source/main
../../build/nbody
```

Simulation flags can be appended to the executable, exactly as with `animation.py` but without the quoting (each flag is its own shell argument here):

```bash
../../build/nbody -dim 2 -simT 0
```

As with `animation.py`, running with no arguments triggers the same interactive `Enter 'd'...` prompt.

## Command-line flags

All flags are optional; any flag not provided falls back to the default shown below. Every numeric flag is validated: an out-of-range or malformed value prints an error and exits with a non-zero status without running anything.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `-h` | — | — | Prints the flag list and exits immediately (ignores every other flag). |
| `-dim <2\|3>` | int | `2` | Number of spatial dimensions. See the note right after this table before combining `-dim 3` with `-simT 2` or `3`. |
| `-simT <0\|1\|2\|3>` | int | `1` | Simulation mode: `0` = serial brute force, `1` = parallel brute force, `2` = serial Barnes-Hut, `3` = parallel Barnes-Hut. Barnes-Hut is 2D-only; see the note below. See also the visualization caveat for `-simT 2` above. |
| `-force <0\|1>` | int | `0` | Interaction law: `0` = gravitational (particle "property" is a mass, always positive), `1` = Coulomb (particle "property" is a charge, generated in the range `[-maxPr, maxPr]`, so both signs occur and like charges repel). |
| `-delta <double>` | double | `0.01` | Time step (delta_t) used by the integrator (Euler for the two brute-force modes, Velocity Verlet for the two Barnes-Hut modes). Must be ≥ 0. Smaller values give a more accurate trajectory at the cost of more iterations to cover the same simulated time. |
| `-simA <int>` | int | `500` | Half-width of the (square) simulation area: particles are generated approximately in `[-simA, simA]` on each axis, and this is also the boundary used for elastic collisions with the domain edge. Must be ≥ 0. |
| `-it <int>` | int | `1000` | Number of simulation iterations (time steps) to run. Must be ≥ 0. |
| `-numP <int>` | int | `50` | Number of particles to generate. Must be ≥ 0. Larger values make the O(n²) brute-force modes noticeably slower than the O(n log n) Barnes-Hut modes; see the performance chapters of the project report for measured numbers. |
| `-maxPr <int>` | int | `1000` | Upper bound on the particle "property": for gravitational force, mass is generated in `[1, maxPr]`; for Coulomb force, charge is generated in `[-maxPr, maxPr]`. Must be ≥ 0. |
| `-maxVel <int>` | int | `5` | Upper bound (in absolute value) on each initial velocity component, generated uniformly in `[-maxVel, maxVel]`. Must be ≥ 0. |
| `-maxR <int>` | int | `10` | Upper bound on particle radius, generated in `[1, maxR]`; radius affects both the drawn circle size in the visualizer and collision detection. Must be ≥ 0. |
| `-soft <double>` | double | `0.7` | Collision-detection tolerance factor: two particles are considered colliding when the squared distance between them is smaller than `soft` times the squared sum of their radii, and it also sets the spatial-hash cell size used to keep collision detection close to linear cost. It is **not** a classical gravitational softening length — it does not appear in the force calculation itself, only in collision handling. Must be ≥ 0. |
| `-spUp <int>` | int | `1` | Output is written only once every `spUp` iterations, instead of every iteration; increasing it reduces file size and I/O time without changing the simulated physics. Must be ≥ 0. |
| `-chunk <int>` | int | `1` | Chunk size for the dynamic OpenMP schedule used by the parallel brute-force solver's force loop (`-simT 1` only; ignored by the other three modes). Must be > 0. See below for how to experiment with it. |
| `-file <path>` | string | `../graphics/Info.txt` | Output path for `Info.txt`. See the `animation.py` caveat above before using this flag together with the visualizer. |

Three points deserve special mention for how they interact with the rest of the program, beyond the table above:

- **`-dim 3` silently does nothing for Barnes-Hut.** `main3DSimulation` only implements `-simT 0` and `-simT 1`. Passing `-dim 3 -simT 2` or `-dim 3 -simT 3` does not print an error: particles are generated, a partial `Info.txt` is written, but no simulation loop ever runs and no `Coordinates_i.txt` file is produced, after which the program exits normally as if nothing were wrong. Use `-simT 2` or `3` only with `-dim 2` (the default).
- **`-chunk`** only affects `-simT 1` (parallel brute force): it controls the chunk size of the `schedule(dynamic, chunkSize)` loop that computes pairwise forces, which is scheduled dynamically because later particles have progressively less work than earlier ones (each particle `i` only computes forces against particles `j > i`). To evaluate its effect, keep every other parameter and `OMP_NUM_THREADS` fixed, and try a few values such as `1`, `4`, `16`, and `64`.
- **`-simT`'s thread count is not controlled directly by this flag.** For any mode other than serial (`-simT 1`, `2`, or `3`), the program automatically requests `min(numP, available hardware threads)` OpenMP threads — so asking for more threads than particles has no effect, and there is no flag to force a *smaller* team than the hardware allows other than setting `OMP_NUM_THREADS` in the environment *before* launching the program (`OMP_NUM_THREADS=2 ../../build/nbody -simT 1 ...`), which the program's automatic thread count still respects as its own upper bound.

## Environment variables

- **`NBODY_NO_IO`** — set to any value to skip writing `Info.txt` and every `Coordinates_i.txt` file. Useful for measuring the simulation's own execution time in isolation from disk I/O; leave it unset when you need the output files, including every time you intend to run `animation.py` afterwards.

## Documentation

You can find a detailed report about the reasoning behind the algorithms, as well as a reference manual with all the code implemented for this project generated using Doxygen.

The detailed report about the algorithms is available in
[`N-Body_Project_Report.pdf`](N-Body_Project_Report.pdf).
