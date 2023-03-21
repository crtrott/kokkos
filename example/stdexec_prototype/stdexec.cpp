//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <cstdio>

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <Kokkos_Graph.hpp>
#include <scheduler.hpp>

template <class Iterator>
struct simple_range {
  Iterator first;
  Iterator last;
};

template <class Iterator>
auto begin(simple_range<Iterator>& rng) {
  return rng.first;
}

template <class Iterator>
auto end(simple_range<Iterator>& rng) {
  return rng.last;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout);

  if (argc < 2) {
    fprintf(stderr, "Usage: %s [<kokkos_options>] <size>\n", argv[0]);
    Kokkos::finalize();
    exit(1);
  }

  const long N = strtol(argv[1], nullptr, 10);

  {
    Kokkos::View<int*, Kokkos::SharedSpace> a("A", N);
    Kokkos::deep_copy(a,3);

    //nvexec::stream_context stream_ctx{};
    //stdexec::scheduler auto sch = stream_ctx.get_scheduler();

    Kokkos::StdExec::inline_scheduler sch;
    stdexec::sender auto snd = stdexec::schedule(sch) | stdexec::bulk(N, KOKKOS_LAMBDA(int i) { if(i<10) printf("Hello From Senders: %i %i\n",i,a(i)); });
    stdexec::sync_wait(std::move(snd));

#if 0
    stdexec::sender auto snd2 = stdexec::schedule(sch) | stdexec::then(KOKKOS_LAMBDA() { return a; }) | stdexec::then(KOKKOS_LAMBDA(Kokkos::View<int*, Kokkos::SharedSpace> b) { return b; })
                                | stdexec::bulk(N, KOKKOS_LAMBDA(int i, Kokkos::View<int*, Kokkos::SharedSpace> b) {printf("Hello From Senders2: %i %i\n",i,b(i));}) | stdexec::split();

    stdexec::sync_wait(snd2);
    stdexec::sync_wait(snd2);
#endif
  }
  Kokkos::finalize();

  return 0;
}
