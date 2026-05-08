# CSC3060 Project 5: C++ Performance Optimization Report

## Overview
This report details the optimization strategies implemented for ten C++ kernels as part of Project 5. The primary objective of this project was to leverage advanced hardware-aware programming techniques to maximize execution speed while preserving strict mathematical correctness. 

---

## 1. Bitwise Operations (`bitwise.cpp`)
**Optimization Strategy: 64-bit Chunk Processing (SWAR)**
- **Concept:** Process multiple 8-bit integers simultaneously by casting the array to 64-bit integer pointers, effectively processing 8 bytes at a time (SIMD within a register).
- **Implementation details:** The kernel interprets chunks of `int8_t` as `uint64_t`. The bitwise AND, OR, and XOR operations are performed natively on 64-bit chunks. A scalar remainder loop handles any leftover elements not aligned to 8 bytes.
- **Why it works:** It increases data throughput by a factor of 8 per instruction, maximizing ALUs and reducing loop overhead. 
- **Reference:** *SIMD Within A Register (SWAR) Techniques*, https://en.wikipedia.org/wiki/SWAR

**Performance Metrics**
- **Laptop (Apple Silicon):** ~119,749 ns (2.08x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 2. Branchless ReLU (`relu.cpp`)
**Optimization Strategy: Branch Elimination**
- **Concept:** Replace the conditional `if (x < 0)` branch with a deterministic arithmetic operation.
- **Implementation details:** Replaced the control-flow branch with `std::max(0.0f, x)`.
- **Why it works:** Modern superscalar processors rely heavily on speculative execution and branch prediction. The `if` statement for random data causes frequent branch mispredictions, forcing pipeline flushes. `std::max` maps directly to hardware conditional-select (like `fmax` on ARM/x86), executing sequentially without stalling the pipeline.
- **Reference:** *Branch Prediction and Branchless Programming*, https://algorithmica.org/en/branchless

**Performance Metrics**
- **Laptop (Apple Silicon):** ~237,312 ns (2.31x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 3. Matrix Multiplication (`matmul.cpp`)
**Optimization Strategy: Loop Tiling (Blocking) & Loop Interchange**
- **Concept:** Reorder nested loops to improve memory access patterns and partition matrices into smaller blocks that fit into the CPU cache.
- **Implementation details:** The standard `i, j, k` loops were modified into a blocked approach. The block size was chosen as `64x64`. The innermost computations operate entirely within these cache-resident blocks.
- **Why it works:** By keeping the working set of the matrix within the L1/L2 cache, we avoid expensive DRAM fetches. Furthermore, iterating over contiguous memory elements (cache lines) fully utilizes hardware prefetchers and spatial locality.
- **Reference:** *Cache Blocking for Matrix Multiplication*, https://en.wikipedia.org/wiki/Loop_nest_optimization

**Performance Metrics**
- **Laptop (Apple Silicon):** ~179,646,339 ns (1.8x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 4. Trace Replay (`trace_replay.cpp`)
**Optimization Strategy: Software Prefetching**
- **Concept:** Anticipate future memory accesses and explicitly instruct the processor to fetch them into cache ahead of time.
- **Implementation details:** Inserted `__builtin_prefetch(&records_data[trace_data[i + 16]], 0, 1)` to fetch the record 16 iterations ahead of the current loop execution.
- **Why it works:** Trace replay involves irregular memory accesses (`records[trace[i]]`). The hardware prefetcher cannot predict these random leaps. Explicit prefetching overlaps the memory latency of future data with the computation of current data.
- **Reference:** *Software Prefetching in C/C++*, https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html

**Performance Metrics**
- **Laptop (Apple Silicon):** ~6,521,847 ns ([LAPTOP_SPEEDUP]x Speedup) *(Note: M1 hardware prefetchers often conflict with software prefetch, metrics may be better on Server)*
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 5. Filter Gradient (`filter_gradient.cpp`)
**Optimization Strategy: Array of Structures (AoS) to Structure of Arrays (SoA)**
- **Concept:** Reorganize the memory layout from AoS (interleaved) to SoA (planar) to improve spatial locality for specific field computations.
- **Implementation details:** The `Pixel` data containing `a` through `i` fields was split. We combined `a, b, c` into a single continuous array, `d, e, f` into another, and `g, h, i` into a third. The kernel was updated to traverse these planar arrays.
- **Why it works:** When computing the gradient, we only need certain fields at a time. The AoS layout forces the cache to load unused data (cache line bloat). The SoA layout ensures that every byte fetched into the cache line is immediately consumed by the vector registers.
- **Reference:** *AoS and SoA Data Layouts*, https://en.wikipedia.org/wiki/AoS_and_SoA

**Performance Metrics**
- **Laptop (Apple Silicon):** ~7,856,641 ns (3.18x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 6. Graph Processing (`graph.cpp`)
**Optimization Strategy: Compressed Sparse Row (CSR) Conversion**
- **Concept:** Flatten linked-list structures into contiguous arrays to eliminate pointer chasing.
- **Implementation details:** The node-pointer graph was converted into two flat vectors: `offsets` and `edges`. The kernel was updated to process neighbors sequentially via `edges[offsets[u]]` to `edges[offsets[u+1]]`.
- **Why it works:** Pointer chasing inherently causes sequential memory stalls and destroys spatial locality. The CSR format ensures edges for a given node are physically adjacent in memory, allowing hardware to stream them perfectly from DRAM to L1 cache.
- **Reference:** *Compressed Sparse Row (CSR) Format*, https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

**Performance Metrics**
- **Laptop (Apple Silicon):** ~14,502,206 ns ([LAPTOP_SPEEDUP]x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 7. Black-Scholes (`blackscholes.cpp`)
**Optimization Strategy: Fast Mathematical Approximations & Single-Precision Arithmetic**
- **Concept:** Avoid slow, high-precision transcendental functions when single-precision variants are acceptable.
- **Implementation details:** Modified `std::exp`, `std::log`, and `std::sqrt` to their single-precision equivalents `expf`, `logf`, and `sqrtf`. Optimized the `CNDF` function algebraically to reduce arithmetic instructions.
- **Why it works:** Standard `std::exp` promotes `float` to `double` and computes a 64-bit mathematically strict result, invoking heavy microcode routines. The `expf` variants stay strictly in 32-bit vector pipelines, cutting latency by more than half.
- **Reference:** *Optimizing Math Functions in C++*, https://developer.arm.com/documentation/101458/0100/Optimize-math-functions

**Performance Metrics**
- **Laptop (Apple Silicon):** ~1,838,312 ns (2.61x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 8. Sparse SpMM (`sparse_spmm.cpp`)
**Optimization Strategy: Loop Unrolling & Register Blocking**
- **Concept:** Reduce loop iteration overhead and maximize register reuse by manually unrolling loops.
- **Implementation details:** Unrolled the inner dense column loop (`n`) by a factor of 8. We process 8 columns simultaneously for a given sparse element.
- **Why it works:** Unrolling exposes more instruction-level parallelism (ILP) to the CPU. It explicitly caches `csr.values[p]` in a register for 8 consecutive multiplications, drastically reducing L1 cache read bandwidth bottlenecks.
- **Reference:** *Register Blocking for Sparse Matrix Vector Multiplication*, https://bebop.cs.berkeley.edu/pubs/nishtala2007-sc-register_blocking.pdf

**Performance Metrics**
- **Laptop (Apple Silicon):** [LAPTOP_TIME] ns ([LAPTOP_SPEEDUP]x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 9. Gated Residual Feature Fusion (`grff.cpp`)
**Optimization Strategy: Loop Fusion & Scalar Replacement**
- **Concept:** Combine multiple independent loops that iterate over the same array to improve temporal locality and eliminate intermediate memory arrays.
- **Implementation details:** Fused 9 separate array traversals into 2 main loops. Intermediate arrays like `G`, `B_prime`, `C_prime`, `H`, and `E` were completely removed, replaced by single scalar variables.
- **Why it works:** Writing to and reading from intermediate memory arrays causes memory bandwidth saturation. By keeping intermediate results (`g`, `c_prime`, `h`) purely in CPU registers, memory traffic is minimized, resulting in a massively compute-bound operation.
- **Reference:** *Loop Fusion Optimization*, https://en.wikipedia.org/wiki/Loop_fission_and_fusion

**Performance Metrics**
- **Laptop (Apple Silicon):** ~1,498,858 ns (5.67x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)

---

## 10. Image Processing (`image_proc.cpp`)
**Optimization Strategy: Function Inlining & Mathematical Simplification**
- **Concept:** Eliminate the overhead of `__attribute__((noinline))` function calls and condense mathematical chains.
- **Implementation details:** Manually inlined all image processing pipeline stages (`color_correct`, `enhance_contrast`, `hdr_compress`, `complex_mask_logic`) directly into the nested loop. Simplified the mathematics to eliminate redundant branches and calculations.
- **Why it works:** The non-inlined functions forced the compiler to emit `call` and `ret` instructions, breaking vectorization and thrashing the stack. Inlining everything explicitly allows the compiler to form one massive dependency graph, resulting in an auto-vectorized master loop.
- **Reference:** *The Benefits of Inline Functions*, https://isocpp.org/wiki/faq/inline-functions

**Performance Metrics**
- **Laptop (Apple Silicon):** ~22,511,189 ns (1.91x Speedup)
- **Server:** [SERVER_TIME] ns ([SERVER_SPEEDUP]x Speedup)
