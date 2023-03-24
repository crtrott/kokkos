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
#include <inline_scheduler.hpp>

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

int threadidx() {
  KOKKOS_IF_ON_HOST( return 999; )
  KOKKOS_IF_ON_DEVICE( return threadIdx.y; )
}
template<class Scheduler, class Data>
KOKKOS_FUNCTION
auto hello_world(Scheduler sch, int N, Data a) {
  return stdexec::schedule(sch)
       | stdexec::bulk(N, KOKKOS_LAMBDA(int i) {
           if(i<10) { printf("HelloWorld: %i %i %i\n",i,a(i), threadidx()); }
         });
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

    Kokkos::View<int[4],Kokkos::SharedSpace> counter("C");
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(2,7,2), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      Kokkos::atomic_add(&counter(0),1);
      Kokkos::single(Kokkos::PerTeam(team), [=]() { Kokkos::atomic_add(&counter(1),1); });
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 10), [=](int ) {
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 10), [=](int ) {
               Kokkos::atomic_add(&counter(2),1); });
         });
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 10), [=](int ) { Kokkos::atomic_add(&counter(3),1); });
    });
    Kokkos::fence();
    printf("Counters: %i %i %i\n",counter(0), counter(1), counter(2));
    //nvexec::stream_context stream_ctx{};
    //stdexec::scheduler auto sch = stream_ctx.get_scheduler();

    //Kokkos::StdExec::inline_scheduler<Kokkos::DefaultExecutionSpace> sch;
#if 0
    Kokkos::StdExec::kokkos_scheduler<Kokkos::DefaultExecutionSpace> sch;
    stdexec::sender auto snd = hello_world(sch, N, a);
    stdexec::sync_wait(std::move(snd));

    
    auto snd2 = stdexec::schedule(sch) 
            | stdexec::bulk(N, KOKKOS_LAMBDA(int i) { 
                if(i<10) {
                  printf("Hello Outer: %i %i %i\n",i,a(i),threadidx()); 
                  Kokkos::StdExec::inline_scheduler<Kokkos::DefaultExecutionSpace> inner_sch;
                  auto snd_inner = hello_world(inner_sch,3,a);
                  stdexec::sync_wait(std::move(snd_inner));
                }
              });

    stdexec::sync_wait(std::move(snd2));
#endif
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(2,4), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      printf("Hello from team: %i %i\n",team.league_rank(), threadidx());
      Kokkos::StdExec::kokkos_scheduler<Kokkos::TeamPolicy<>::member_type> sch(team);
        stdexec::sender auto snd = hello_world(sch, N, a);
//      stdexec::sync_wait(std::move(snd));
      auto op = stdexec::connect(std::move(snd), Kokkos::StdExec::impl_kokkos_scheduler::sync_wait_receiver(sch));
      stdexec::start(op);
    });
    Kokkos::fence();
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
