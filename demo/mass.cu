/*
Siggraph Asia 2012 Demo

Mass vector implementation

Laurence Emms
*/

#include <iostream>
#include <vector>
#include <cuda.h>
#include "mass.h"

SigAsiaDemo::Mass::Mass(
	float mass,
	float x,
	float y,
	float z,
	float vx,
	float vy,
	float vz,
	float ax,
	float ay,
	float az) :
		_mass(mass),
		_x(x),
		_y(y),
		_z(z),
		_vx(vx),
		_vy(vy),
		_vz(vz),
		_ax(ax),
		_ay(ay),
		_az(az)
{}

SigAsiaDemo::MassList::MassList() :
	_computing(false),
	_changed(false),
	_device_masses(0)
{}

SigAsiaDemo::MassList::~MassList()
{
	if (_device_masses) {
		cudaFree(_device_masses);
	}
}

bool SigAsiaDemo::MassList::push(Mass mass)
{
	// enforce that no masses can be added
	// if the buffer is uploaded to the GPU
	if (!_computing) {
		_masses.push_back(mass);
		_changed = true;
		return true;
	}
	return false;
}

size_t SigAsiaDemo::MassList::size() const
{
	return _masses.size();
}

void SigAsiaDemo::MassList::upload()
{
	if (_changed) {
		std::cout << "Upload masses" << std::endl;
		_changed = false;
		if (_device_masses) {
			std::cout << "Free masses." << std::endl;
			cudaFree(_device_masses);
		}

		// allocate GPU buffer
		std::cout << "Allocate GPU buffer of size " << _masses.size() << "." << std::endl;
		cudaMalloc(
			(void**)&_device_masses,
			_masses.size()*sizeof(Mass));

		// copy into GPU buffer
		std::cout << "Copy masses into GPU buffer." << std::endl;
		cudaMemcpy(
			_device_masses,
			&_masses[0],
			_masses.size()*sizeof(Mass),
			cudaMemcpyHostToDevice);
	}

	_computing = true;
}

void SigAsiaDemo::MassList::download()
{
	if (_changed) {
		std::cerr << "Error: Mass list changed while \
data was being used in GPU computations." << std::endl;
		std::terminate();
	} else {
		// copy into CPU buffer
		std::cout << "Copy masses into CPU buffer." << std::endl;
		cudaMemcpy(
			&_masses[0],
			_device_masses,
			_masses.size()*sizeof(Mass),
			cudaMemcpyDeviceToHost);
	}
	_computing = false;
}

SigAsiaDemo::Mass *SigAsiaDemo::MassList::getMass(size_t index)
{
	if (_masses.empty()) {
		std::cout << "Warning: getMass called on \
empty mass list." << std::endl;
		return 0;
	}
	if (_computing) {
		std::cout << "Warning: getMass called while \
mass list is uploaded to the GPU." << std::endl;
		return 0;
	}
	if (index >= _masses.size()) {
		std::cout << "Warning: getMass called on index \
out of bounds." << std::endl;
		return 0;
	}

	return &_masses[index];
}

SigAsiaDemo::Mass *SigAsiaDemo::MassList::getDeviceMasses()
{
	return _device_masses;
}

__global__ void deviceUpdate(float dt, int N, SigAsiaDemo::Mass *masses)
{
	// add gravity
	int tid = blockIdx.x;
	if (tid < N) {
		// TODO: RK2/4
		masses[tid]._x += masses[tid]._vx * dt;
		masses[tid]._y += masses[tid]._vy * dt;
		masses[tid]._z += masses[tid]._vz * dt;
		masses[tid]._vx += masses[tid]._ax * dt * dt;
		masses[tid]._vy += masses[tid]._ay * dt * dt;
		masses[tid]._vz += masses[tid]._az * dt * dt;
		masses[tid]._az = 9.8 / masses[tid]._mass;
	}
}

void SigAsiaDemo::MassList::update(float dt)
{
	if (_computing) {
		std::cout << "Update masses (" << _masses.size() << ")." << std::endl;
		deviceUpdate<<<_masses.size(), 1>>>(dt, _masses.size(), _device_masses);
	}
}
