# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
LDFLAGS := -lmpi
INCLUDES = -Ilib

TARGET = DcuQC

$(TARGET): main.o common.o host_general.o host_expansion.o host_helper.o host_debug.o device_general.o device_expansion.o device_helper.o device_debug.o Quick_rmnonmax.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cpp common.h host_general.h host_debug.h Quick_rmnonmax.h
	$(NVCC) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

common.o: common.cpp common.h
	$(CXX) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

host_general.o: host_general.cpp common.h host_general.h host_expansion.h host_helper.h host_debug.h device_general.h
	$(NVCC) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

host_expansion.o: host_expansion.cpp common.h host_expansion.h host_helper.h host_debug.h
	$(CXX) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

host_helper.o: host_helper.cpp common.h host_debug.h
	$(CXX) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

host_debug.o: host_debug.cpp common.h host_debug.h
	$(CXX) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

device_general.o: device_general.cpp common.h device_general.h device_expansion.h device_helper.h device_debug.h
	$(NVCC) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

device_expansion.o: device_expansion.cpp common.h device_expansion.h device_helper.h
	$(NVCC) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

device_helper.o: device_helper.cpp common.h device_helper.h device_debug.h
	$(NVCC) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

device_debug.o: device_debug.cpp common.h device_debug.h
	$(NVCC) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

Quick_rmnonmax.o: Quick_rmnonmax.cpp common.h Quick_rmnonmax.h
	$(CXX) $(INCLUDES) -c $< -o $@ $(LDFLAGS)