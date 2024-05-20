# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -std=c++11 -arch=sm_70
CXXFLAGS = -O3 -std=c++11
LDFLAGS := -lmpi
INCLUDES = -Iinc

SOURCES = main.cu src/common.cpp src/host_general.cu src/host_expansion.cpp src/host_helper.cpp src/host_debug.cpp src/device_general.cu src/device_expansion.cu src/device_helper.cu src/device_debug.cu src/Quick_rmnonmax.cpp
OBJECTS = main.o src/common.o src/host_general.o src/host_expansion.o src/host_helper.o src/host_debug.o src/device_general.o src/device_expansion.o src/device_helper.o src/device_debug.o src/Quick_rmnonmax.o
TARGET = DcuQC

$(TARGET): $(OBJECTS)
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cu inc/common.h inc/host_general.h inc/host_debug.h inc/Quick_rmnonmax.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/common.o: src/common.cpp inc/common.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/host_general.o: src/host_general.cu inc/common.h inc/host_general.h inc/host_expansion.h inc/host_helper.h inc/host_debug.h inc/device_general.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/host_expansion.o: src/host_expansion.cpp inc/common.h inc/host_expansion.h inc/host_helper.h inc/host_debug.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/host_helper.o: src/host_helper.cpp inc/common.h inc/host_debug.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/host_debug.o: src/host_debug.cpp inc/common.h inc/host_debug.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

src/device_general.o: src/device_general.cu inc/common.h inc/device_general.h inc/device_expansion.h inc/device_helper.h inc/device_debug.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/device_expansion.o: src/device_expansion.cu inc/common.h inc/device_expansion.h inc/device_helper.h inc/device_debug.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/device_helper.o: src/device_helper.cu inc/common.h inc/device_helper.h inc/device_debug.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/device_debug.o: src/device_debug.cu inc/common.h inc/device_debug.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/Quick_rmnonmax.o: src/Quick_rmnonmax.cpp inc/common.h inc/Quick_rmnonmax.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm $(OBJECTS) $(TARGET)