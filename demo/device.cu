/*
Siggraph Asia 2012 Demo

CUDA device implementation.

Laurence Emms
*/

#include <iostream>

// CUDA
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "device.h"

int SigAsiaDemo::setDevice(int &device)
{
	// get cuda device properties
	cudaDeviceProp properties;
	int count;
	cudaGetDeviceCount(&count);
	if (count <= 0) {
		std::cerr << "Error: No CUDA devices found." << std::endl;
		return 1;
	}
	int max_major = 0;
	int max_minor = 0;
	for (int i = 0; i < count; ++i) {
		std::cout << "CUDA Device " << i << ":" << std::endl;
		cudaGetDeviceProperties(&properties, i);
		std::cout << "Name: " << properties.name << std::endl;
		std::cout << "Compute Capability: " \
		<< properties.major << "." << properties.minor << std::endl;
		std::cout << "Global Memory: " \
		<< properties.totalGlobalMem << std::endl;
		std::cout << "Constant Memory: " \
		<< properties.totalConstMem << std::endl;
		std::cout << "Shared Memory: " \
		<< properties.sharedMemPerBlock << std::endl;
		std::cout << "Registers: " \
		<< properties.regsPerBlock << std::endl;
		std::cout << "Warp Size: " \
		<< properties.warpSize << std::endl;
		std::cout << "Max Threads Per Block: " \
		<< properties.maxThreadsPerBlock << std::endl;
		std::cout << "Max Threads: (" \
		<< properties.maxThreadsDim[0] << ", " \
		<< properties.maxThreadsDim[1] << ", " \
		<< properties.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Max Grid Size: (" \
		<< properties.maxGridSize[0] << ", " \
		<< properties.maxGridSize[1] << ", " \
		<< properties.maxGridSize[2] << ")" << std::endl;

		// get the device with the highest compute cabability
		if (properties.major > max_major) {
			device = i;
			max_major = properties.major;
			max_minor = properties.minor;
		} else if (	properties.major == max_major &&
					properties.minor > max_minor) {
			device = i;
			max_major = properties.major;
			max_minor = properties.minor;
		}
	}
	std::cout << "CUDA Device " << device << " chosen." << std::endl;
	if (max_major < 1) {
		std::cerr << "Error: CUDA version >= 1.0 required." << std::endl;
		return 1;
	}

	// set device
	cudaSetDevice(device);

	return 0;
}

void SigAsiaDemo::setGLDevice(int device)
{
	cudaGLSetGLDevice(device);
	return;
}
