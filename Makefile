# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
# NOTE - -O optimization flags cause bugs, don't use them
NVCCFLAGS = -gencode arch=compute_80,code=sm_80 -std=c++11
CXXFLAGS = -std=c++11
LDFLAGS := -lmpi
INCLUDES = -Iinc

OBJECTS1 = src/common.o src/host_debug.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o program1.o
TARGET1 = program1

OBJECTS2 = src/common.o src/host_debug.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o program2.o
TARGET2 = program2

OBJECTS3 = src/common.o src/Quick_rmnonmax.o program3.o
TARGET3 = program3

all: d2u $(TARGET1) $(TARGET2) $(TARGET3)

.PHONY: d2u
d2u:
	dos2unix *.sh

$(TARGET1): $(OBJECTS1)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(TARGET2): $(OBJECTS2)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(TARGET3): $(OBJECTS3)
	$(NVCC) $^ -o $@ $(LDFLAGS)

program1.o: program1.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

program2.o: program2.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

program3.o: program3.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/common.o: src/common.cpp inc/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/host_functions.o: src/host_functions.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/device_kernels.h inc/cuTS_MPI.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/host_debug.o: src/host_debug.cpp inc/common.h inc/host_debug.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/device_kernels.o: src/device_kernels.cu inc/common.h inc/device_kernels.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/Quick_rmnonmax.o: src/Quick_rmnonmax.cpp inc/common.h inc/Quick_rmnonmax.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/cuTS_MPI.o: src/cuTS_MPI.cpp inc/common.h inc/cuTS_MPI.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: c
c: ct co cp

.PHONY: ct
ct:
	rm -f $(OBJECTS1) $(OBJECTS2) e_* t_* r_* s_*

.PHONY: co
co:
	rm -f o_*

.PHONY: cp
cp:
	rm -f $(TARGET1) $(TARGET2)