// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

/*
 * Atomic operations microbenchmark — Kokkos RangePolicy.
 *
 * Kernel: for each node v, iterate neighbors and perform
 * atomic_load, atomic_fetch_add, atomic_compare_exchange, atomic_store.
 *
 * Usage: ./atomic_range <input.egr>
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using View1D = Kokkos::View<int32_t*, Kokkos::LayoutLeft>;

struct AtomicBenchRange {
    int32_t nodes;
    View1D nidx, nlist, data;

    KOKKOS_INLINE_FUNCTION
    void operator()(int32_t v) const {
        const int32_t beg = nidx(v);
        const int32_t end = nidx(v + 1);
        for (int32_t i = beg; i < end; ++i) {
            const int32_t nli = nlist(i);
            int32_t old = Kokkos::atomic_load(&data(nli));
            Kokkos::atomic_fetch_add(&data(nli), 1);
            Kokkos::atomic_compare_exchange(&data(nli), old + 1, old + 2);
            Kokkos::atomic_store(&data(v), old);
        }
    }
};

struct AtomicBenchRangeCuda {
    int32_t nodes;
    View1D nidx, nlist, data;

    KOKKOS_INLINE_FUNCTION
    void operator()(int32_t v) const {
        const int32_t beg = nidx(v);
        const int32_t end = nidx(v + 1);
        for (int32_t i = beg; i < end; ++i) {
            const int32_t nli = nlist(i);
#ifdef __CUDA_ARCH__
            //int32_t old = atomicRead(&data(nli));
            int32_t old;
            __nv_atomic_load(&data(nli), &old, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            atomicAdd(&data(nli), 1);
            atomicCAS(&data(nli), old + 1, old + 2);
            //atomicWrite(&data(v), old);
            __nv_atomic_store(&data(v), &old, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
#endif
        }
    }
};

void test() {
        const int32_t N = 1000000;
        const int32_t M = 5 * N;
        std::cout << "Atomic benchmark: Kokkos RangePolicy\n";
        std::cout << "input graph: " << N << " nodes and " << M << " edges\n";

        View1D nidx(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nidx"), N + 1);
        View1D nlist(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nlist"), M);
        Kokkos::Random_XorShift64_Pool<> rand_pool64(5374857);
        Kokkos::parallel_for("set counts", N+1, KOKKOS_LAMBDA(int i) { nidx(i) = 5*i; });
        Kokkos::fill_random(nlist, rand_pool64, N);
        View1D data("data", N);

        Kokkos::parallel_for("atomic_range", Kokkos::RangePolicy<>(0, N),
                             AtomicBenchRange{N, nidx, nlist, data});
        Kokkos::parallel_for("atomic_range_cuda", Kokkos::RangePolicy<>(0, N),
                             AtomicBenchRangeCuda{N, nidx, nlist, data});

        Kokkos::fence();
        // Timed run
        {
        Kokkos::Timer timer;
        Kokkos::parallel_for("atomic_range", Kokkos::RangePolicy<>(0, N),
                             AtomicBenchRange{N, nidx, nlist, data});
        Kokkos::fence();
        double runtime = timer.seconds();

        std::cout << "kernel time: " << runtime << " s\n";
        std::cout << "throughput:  " << (M * 1e-6 / runtime) << " Matomics/s\n";
        }
        {
        Kokkos::Timer timer;
        Kokkos::parallel_for("atomic_range_cuda", Kokkos::RangePolicy<>(0, N),
                             AtomicBenchRangeCuda{N, nidx, nlist, data});
        Kokkos::fence();
        double runtime = timer.seconds();

        std::cout << "kernel time: " << runtime << " s\n";
        std::cout << "throughput:  " << (M * 1e-6 / runtime) << " Matomics/s\n";
        }
}
namespace Test {

TEST(defaultdevicetype, development_test) { test(); }

}  // namespace Test
