



    Kokkos::parallel_for(Kokkos::TeamPolicy<>(N,7,2), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      Kokkos::atomic_add(&counter(0),1);
      Kokkos::single(Kokkos::PerTeam(team), [=]() { Kokkos::atomic_add(&counter(1),1); });
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 10), [=](int ) {
         Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 10), [=](int ) {
               Kokkos::atomic_add(&counter(2),1); });
         });
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, 10), [=](int ) { Kokkos::atomic_add(&counter(3),1); });
    });

hierarchical_scheduler<3> sched(AUTO,AUTO);

template<int Rank, class Scheduler, int Total = Rank>
class hierarchical_scheduler {
  using sub_scheduler = inline_scheduler;
  //hierachical_scheduler<Rank-1>;
};


template<int Rank>
class hierarchical_scheduler<Rank, Kokkos::ExecSpaceScheduler, Rank> {
  using sub_scheduler = Kokkos::OuterTeamScheduler;
  //hierachical_scheduler<Rank-1>;
};

template<int Rank>
class hierarchical_scheduler<Rank, Kokkos::ExecSpaceScheduler, Rank+1> {
  using sub_scheduler = Kokkos::ThreadScheduler;
  //hierachical_scheduler<Rank-1>;
};

template<int Rank>
class hierarchical_scheduler<Rank, Kokkos::ExecSpaceScheduler, Rank+2> {
  using sub_scheduler = Kokkos::VectorScheduler;
  //hierachical_scheduler<Rank-1>;
};

template<int Rank, int Total>
class hierarchical_scheduler<Rank, Kokkos::ExecSpaceScheduler,Total> {
  using sub_scheduler = inline_scheduler;
  //hierachical_scheduler<Rank-1>;
};
auto snd = schedule(sched) | bulk_concurrent_nested(nest<2>(4,8), N, [=](auto sub_scheduler, int i) {
  
  auto snd = schedule(sub_scheduler)
  | bulk(M, [=](int j) {
    printf("%i %i\n",i,j);
  })
  | reduce(M, [=](int j) {
  });

  auto val = sync_wait(snd).get_value();
  val = exp(val);
  auto snd = schedule(sub_scheduler)
  | bulk(M, [=](int j) {
    printf("%i %i %i\n",i,j,val);
  });

  return snd;
});

parallel_for(TeamPolicy<>(N,4) [=](auto team) {
  int i = team.league_rank();
  parallel_for(RangePolicy(team, M), [=](int j) {
  });
  team.team_barrier();
});
auto snd = schedule(sched) | bulk_concurrent_nested(nest<1>(4), N, [=](auto sub_scheduler, int i) {
  sync_wait(schedule(sub_scheduler) | bulk(M, [=](int j) {
  }));
});


parallel_for(TeamPolicy<>(N,4,8) [=](auto team) {
  int i = team.league_rank();
  parallel_for(TeamThreadPolicy(team, M), [=](int j) {
    parallel_for(RangePolicy(team, K), [=](int k) {
    });
  });
  parallel_for(RangePolicy(team, M), [=](int j) {
  });
  team.team_barrier();
});

auto snd = schedule(sched) | bulk_concurrent_nested(nest<2>(4,8), N, [=](auto sub_scheduler, int i) {
  auto sndnested1 = schedule(sub_scheduler) | bulk_conncurrent_nested(nest<1>(8), M, [=](auto sub2, int j) {
    auto sndnested2 = schedule(sub2) | bulk(K, [=](int k) {
    });
    sync_wait(sndnested2);
  }))
  | bulk(M, [=](int j) {});
  sync_wait(sndnested1);
});


matrix_vector_product(scheduler sched, mdspan A, mdspan x, mdspan y) {
  for(int i=0; i<A.extent(0); i++) {
    y[i] = 0;
    for(int j=0; j<A.extent(1); j++)
      y[i] += A[i,j]*x[j];
  }
}


matrix_vector_product(scheduler sched, matrix A, vector x, vector y) {
   int nested_concurrency = get_preferred_nested_concurrency<1>(sched);
   auto hsched = get_hierarchical_scheduler<2>(sched, nested_concurrency);
   auto snd = schedule(hsched) |
      bulk_concurrent_nested(A.extent(0), [=](auto sub_scheduler, int i) {
        auto snd = schedule(sub_scheduler) |
           bulk_reduce(A.extent(1), [=](int j) {
             return A[i,j]*x[j];
           }) |
           then([=](auto val) {y[i] = val;});
        sync_wait(snd);
      });
}


matrix_vector_product(scheduler sched, batched_matrix A, batched_vector x, batched_vector y) {
   auto nested_concurrency = get_preferred_nested_concurrency<2>(sched);
   auto hsched = get_hierarchical_scheduler<3>(sched, nested_concurrency);
   auto snd = schedule(hsched) |
      bulk_concurrent_nested(bA.extent(0), [=](auto sub_scheduler, int i) {
        matrix_vector_product(sub_scheduler, submdspan(bA, i,full_extent,full_extent), ...);
      });
}

mdarray<double> A(N,M);
mdarray<double> x(M), y(N);

matrix_vector_product(sched A,x,y);


mdarray<double> bA(K,N,M);
mdarray<double> bx(K,M), by(K,N);

matrix_vector_product(sched, bA, bx, by);
