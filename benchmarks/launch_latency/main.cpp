#include<Kokkos_Core.hpp>

template<int V>
struct TestFunctor {
  double values[V];
  Kokkos::View<double*> a;
  int R;
  TestFunctor(Kokkos::View<double*> a_, int R_):a(a_),R(R_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for(int j=0; j<R;j++)
      a(i) += 1.0*i*values[j];
  }
};


void test_fence() {
  Kokkos::fence();
  //Kokkos::DefaultExecutionSpace().fence();
  //hipDeviceSynchronize();
}
template<int V>
void run(int N, int M, int R) {

  double result;
  Kokkos::View<double*> a("A",N);
  Kokkos::View<double> v_result("result");
  TestFunctor<V> f(a,R);

  // WarmUp
  for(int i=0; i<1000; i++) {
    Kokkos::parallel_for("WarmUp",N,f);
    test_fence();
  }


  // Timing batched
  Kokkos::Timer timer;
  for(int i=0; i<M; i++) {
    Kokkos::parallel_for("Test1",N,f);
  }
  double time_batched = timer.seconds();
  timer.reset();
  test_fence();
  double time_fence = timer.seconds();
  timer.reset();

  // Timing single
  for(int i=0; i<M; i++) {
    Kokkos::parallel_for("Test2",N,f);
    test_fence();
  }
  double time_single = timer.seconds();

  double x = 1.e6/M;
  printf("N %i M %i R %i V %i parallel_for: batched: %lf single: %lf fence: %e\n",
      N,M,R,V,
      x*time_batched,x*time_single,time_fence
      );

}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
     int N = (argc>1) ? atoi(argv[1]) : 10000;
     int M = (argc>2) ? atoi(argv[2]) : 10000;
     int R = (argc>3) ? atoi(argv[3]) : 10;
    
     run<1>(N,M,R<=1?R:1);
     run<16>(N,M,R<=16?R:16);
     run<200>(N,M,R<=200?R:200);
     run<3000>(N,M,R<=3000?R:3000);
     run<30000>(N,M,R<=30000?R:30000);
  }
  Kokkos::finalize();
}
