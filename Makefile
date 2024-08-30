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

OBJECTS0 = program0.o
TARGET0 = program0

OBJECTS1 = src/common.o src/host_debug.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o program1.o
TARGET1 = program1

OBJECTS2 = src/common.o src/host_debug.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o program2.o
TARGET2 = program2

OBJECTS3 = src/common.o src/Quick_rmnonmax.o program3.o
TARGET3 = program3

all: d2u $(TARGET0) $(TARGET1) $(TARGET2) $(TARGET3)

.PHONY: d2u
d2u:
	dos2unix *.sh

$(TARGET0): $(OBJECTS0)
	$(CXX) $^ -o $@ $(CXXLDFLAGS)

$(TARGET1): $(OBJECTS1)
	$(NVCC) $^ -o $@ $(NVCCLDFLAGS)

$(TARGET2): $(OBJECTS2)
	$(NVCC) $^ -o $@ $(NVCCLDFLAGS)

$(TARGET3): $(OBJECTS3)
	$(CXX) $^ -o $@ $(CXXLDFLAGS)

program0.o: program0.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

program1.o: program1.cu inc/common.hpp inc/host_functions.hpp inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

program2.o: program2.cu inc/common.hpp inc/host_functions.hpp inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

program3.o: program3.cpp inc/common.hpp inc/host_functions.hpp inc/host_debug.h inc/Quick_rmnonmax.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

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

.PHONY: c
c: ct co cp

.PHONY: ct
ct:
	rm -f $(OBJECTS0) $(OBJECTS1) $(OBJECTS2) $(OBJECTS3) e_* t_* r_* s_*

.PHONY: co
co:
	rm -f o_*

.PHONY: cp
cp:
	rm -f $(TARGET0) $(TARGET1) $(TARGET2) $(TARGET3)