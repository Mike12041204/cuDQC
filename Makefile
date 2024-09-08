# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
# NOTE - -O optimization flags cause bugs, don't use them
NVCCFLAGS = -gencode arch=compute_80,code=sm_80 -std=c++11 -Xcompiler "-fopenmp"
CXXFLAGS = -std=c++11 -fopenmp
NVCCLDFLAGS := -lmpi -Xcompiler "-fopenmp"
CXXLDFLAGS := -lmpi -fopenmp
INCLUDES = -Iinc

OBJECTS = program.o src/common.o src/host_functions.o src/Quick_rmnonmax.o src/host_debug.o src/device_kernels.o src/cuTS_MPI.o
TARGET = cuDQC

all: d2u $(TARGET)

.PHONY: d2u
d2u:
	dos2unix *.sh

$(TARGET): $(OBJECTS)
	$(NVCC) $^ -o $@ $(NVCCLDFLAGS)

program.o: program.cu inc/common.hpp inc/host_functions.hpp inc/Quick_rmnonmax.h inc/host_debug.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/common.o: src/common.cpp inc/common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/host_functions.o: src/host_functions.cu inc/common.hpp inc/host_functions.hpp inc/host_debug.h inc/device_kernels.hpp inc/cuTS_MPI.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/host_debug.o: src/host_debug.cpp inc/common.hpp inc/host_debug.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/device_kernels.o: src/device_kernels.cu inc/common.hpp inc/device_kernels.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/Quick_rmnonmax.o: src/Quick_rmnonmax.cpp inc/common.hpp inc/Quick_rmnonmax.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/cuTS_MPI.o: src/cuTS_MPI.cpp inc/common.hpp inc/cuTS_MPI.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# various clean and print shortcuts
.PHONY: c
c: ct co

.PHONY: cc
cc: c cp

.PHONY: ct
ct:
	rm -f $(OBJECTS) DQC-E* DQC-T* DQC-R*

.PHONY: co
co:
	rm -f DQC-O*

.PHONY: cp
cp:
	rm -f $(TARGET)

.PHONY: p
p:
	cat DQC-O_*

.PHONY: pp
pp:
	cat DQC-O*