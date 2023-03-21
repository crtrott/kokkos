# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.16)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Example CXX)

find_package(STDEXEC REQUIRED)

# We build kokkos as a subdirectory of our project
add_subdirectory(${Example_SOURCE_DIR}/../.. ${Example_BINARY_DIR}/kokkos)

add_executable(kokkos-stdexec stdexec.cpp)
target_include_directories(kokkos-stdexec PRIVATE ${Example_SOURCE_DIR})

# This is the only line required to set up all compiler/linker flags
target_link_libraries(kokkos-stdexec Kokkos::kokkos STDEXEC::stdexec)

# Adds a test for the executable
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
