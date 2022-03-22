#include <Kokkos_Core.hpp>

template <class Scalar, class DeviceType>
void test_deep_copy(int N, int R) {

  Kokkos::View<Scalar*, DeviceType> v_d("v", N);
  auto v_h = Kokkos::create_mirror_view(v_d);
  Kokkos::fence();
 
  size_t size = N * sizeof(Scalar);

  //warmup
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
        KOKKOS_LAMBDA(const int i) { v_h(i) = 0; });

  Kokkos::Timer timer;
  for (int r = 0; r < R; r++) {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
        KOKKOS_LAMBDA(const int i) { v_h(i) = 0; });
  }
  Kokkos::fence();
  double time_host_fill = timer.seconds();

  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;
  timer.reset();
  for (int r = 0; r < R; r++) {
    Kokkos::deep_copy(exec_space, v_d, v_h);
  }
  Kokkos::fence();
  double time_copy_to_device = timer.seconds();

  timer.reset();
  for (int r = 0; r < R; r++) {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec_space, 0, N),
        KOKKOS_LAMBDA(const int i) { ++v_d(i); });
  }
  Kokkos::fence();
  double time_increment_on_device = timer.seconds();

  timer.reset();
  for (int r = 0; r < R; r++) {
    Kokkos::deep_copy(v_h, v_d);
  }
  Kokkos::fence();
  double time_copy_to_host = timer.seconds();

  timer.reset();
  for (int r = 0; r < R; r++) {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
        KOKKOS_LAMBDA(const int i) { v_h(i) = 0; });
  }
  double time_host_reset = timer.seconds();

  printf("fill on host        %f \n",     R * size / time_host_fill           / 1024 / 1024 / 1024);
  printf("copy host to device %f \n",     R * size / time_copy_to_device      / 1024 / 1024 / 1024);
  printf("increment on device %f \n", 2 * R * size / time_increment_on_device / 1024 / 1024 / 1024);
  printf("copy device to host %f \n",     R * size / time_copy_to_host        / 1024 / 1024 / 1024);
  printf("reset on host       %f \n",     R * size / time_host_reset          / 1024 / 1024 / 1024);
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    if (argc < 4) {
      printf("Arguments: N R T\n");
      printf("  N:   Length of array to do atomics into\n");
      printf("  R:   Number of repeats of the experiments\n");
      printf("  T:   Type of atomic\n");
      printf("       1 - int\n");
      printf("       2 - long\n");
      printf("       3 - float\n");
      printf("       4 - double\n");
      printf("Example Input GPU:\n");
      printf("  Histogram : 1000000 1000 1 1000 1 10 1\n");
      printf("  MD Force : 100000 100000 100 1000 20 10 4\n");
      printf("  Matrix Assembly : 100000 1000000 50 1000 20 10 4\n");
      Kokkos::finalize();
      return 0;
    }

    int N    = std::stoi(argv[1]);
    int R    = std::stoi(argv[2]);
    int type = std::stoi(argv[3]);

    //using DeviceType = Kokkos::DefaultExecutionSpace;
    using DeviceType = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>;
    //using DeviceType = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;

    double time = 0;
    if (type == 1) test_deep_copy<int, DeviceType>(N, R);
    if (type == 2) test_deep_copy<long, DeviceType>(N, R);
    if (type == 3) test_deep_copy<float, DeviceType>(N, R);
    if (type == 4) test_deep_copy<double, DeviceType>(N, R);

    double time2 = 1;
    int size = 0;
    if (type == 1) size = sizeof(int);
    if (type == 2) size = sizeof(long);
    if (type == 3) size = sizeof(float);
    if (type == 4) size = sizeof(double);

    printf("%i\n", size);
    printf(
        "Time: %s %i %i (t_atomic: %e t_nonatomic: %e ratio: %lf "
        ")( GUpdates/s: %lf GB/s: %lf )\n",
        (type == 1)
            ? "int"
            : ((type == 2)
                   ? "long"
                   : ((type == 3) ? "float"
                                  : "double")),
        N, R, time, time2, time / time2, 1.e-9 * R / time,
        1.0 * R * 2 * size / time / 1024 / 1024 / 1024);
  }
  Kokkos::finalize();
}
