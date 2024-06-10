# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -gencode arch=compute_80,code=sm_80 -std=c++11
CXXFLAGS = -O3 -std=c++11
LDFLAGS := -lmpi
INCLUDES = -Iinc

OBJECTS1 = src/common.o src/host_debug.o src/Quick_rmnonmax.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o program1.o
TARGET1 = program1

OBJECTS2 = src/common.o src/host_debug.o src/Quick_rmnonmax.o src/cuTS_MPI.o src/device_kernels.o src/host_functions.o program2.o
TARGET2 = program2

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJECTS1)
	$(NVCC) $^ -o $@ $(LDFLAGS)

$(TARGET2): $(OBJECTS2)
	$(NVCC) $^ -o $@ $(LDFLAGS)

program1.o: program1.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

program2.o: program2.cu inc/common.h inc/host_functions.h inc/host_debug.h inc/Quick_rmnonmax.h
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

.PHONY: clean
clean:
	rm -f $(OBJECTS1) $(TARGET1) $(OBJECTS2) $(TARGET2) error_DcuQC.txt results_DcuQC.txt output_DcuQC* temp_DcuQC*