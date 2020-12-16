KOKKOS_DEVICES = HIP
KOKKOS_CUDA_OPTIONS=enable_lambda
KOKKOS_ARCH = Vega900
KOKKOS_HIP_KL_STATIC = 0
KOKKOS_HIP_KL_HP = 0

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

ifndef KOKKOS_PATH
  KOKKOS_PATH = ${HOME}/Kokkos/kokkos
endif

SRC = $(wildcard $(MAKEFILE_PATH)*.cpp)
HEADERS = $(wildcard $(MAKEFILE_PATH)*.hpp)

vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo Start Build [$(KOKKOS_HIP_KL_STATIC)]

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = test.cuda
else
ifneq (,$(findstring HIP,$(KOKKOS_DEVICES)))
CXX = hipcc
EXE = test.hip
else
CXX = g++
EXE = test.host
endif
endif


CXXFLAGS ?= -O3 -g
override CXXFLAGS += -I$(MAKEFILE_PATH)

ifeq ($(KOKKOS_HIP_KL_STATIC), 1)
override CXXFLAGS += -DKOKKOS_IMPL_HIP_KERNEL_LAUNCH_STATIC
endif
ifeq ($(KOKKOS_HIP_KL_HP), 1)
override CXXFLAGS += -DKOKKOS_IMPL_HIP_KERNEL_LAUNCH_HOST_PINNED
endif

DEPFLAGS = -M
LINK = ${CXX}
LINKFLAGS =

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
