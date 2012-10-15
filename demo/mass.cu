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
	float fx,
	float fy,
	float fz) :
		_mass(mass),
		_x(x), _y(y), _z(z),
		_tx(x), _ty(y), _tz(z),
		_fx(fx), _fy(fy), _fz(fz),
		_k1x(0.0), _k1y(0.0), _k1z(0.0),
		_k2x(0.0), _k2y(0.0), _k2z(0.0),
		_k3x(0.0), _k3y(0.0), _k3z(0.0),
		_k4x(0.0), _k4y(0.0), _k4z(0.0)
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
		std::cout << "Allocate GPU buffer of size " << \
		_masses.size() << "." << std::endl;
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
		std::cout << "Download masses" << std::endl;
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

__global__ void deviceStartFrame(int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		// set temporary position for k1
		masses[tid]._tx = masses[tid]._x;
		masses[tid]._ty = masses[tid]._y;
		masses[tid]._tz = masses[tid]._z;
	}
}

void SigAsiaDemo::MassList::startFrame()
{
	if (_computing) {
		std::cout << "Start frame (" \
		<< _masses.size() << ")." << std::endl;
		deviceStartFrame<<<_masses.size(), 1>>>(
			_masses.size(),
			_device_masses);
	}
}

__global__ void deviceClearForces(int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		masses[tid]._fx = 0.0f;
		// add gravity
		masses[tid]._fy = -9.81f * masses[tid]._mass;
		masses[tid]._fz = 0.0f;
	}
}

void SigAsiaDemo::MassList::clearForces()
{
	if (_computing) {
		std::cout << "Clear forces and add gravity (" \
		<< _masses.size() << ")." << std::endl;
		deviceClearForces<<<_masses.size(), 1>>>(
			_masses.size(),
			_device_masses);
	}
}

__global__ void deviceEvaluateK1(float dt, int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k1
		masses[tid]._k1x += ax * dt;
		masses[tid]._k1y += ay * dt;
		masses[tid]._k1z += az * dt;

		// set temporary position for k2
		masses[tid]._tx = masses[tid]._x + masses[tid]._k1x * dt * 0.5f;
		masses[tid]._ty = masses[tid]._y + masses[tid]._k1y * dt * 0.5f;
		masses[tid]._tz = masses[tid]._z + masses[tid]._k1z * dt * 0.5f;
	}
}

void SigAsiaDemo::MassList::evaluateK1(float dt)
{
	if (_computing) {
		std::cout << "Evaluate K1 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK1<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
	}
}

__global__ void deviceEvaluateK2(float dt, int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k2
		masses[tid]._k2x += ax * dt;
		masses[tid]._k2y += ay * dt;
		masses[tid]._k2z += az * dt;

		// set temporary position for k3
		masses[tid]._tx = masses[tid]._x + masses[tid]._k2x * dt * 0.5f;
		masses[tid]._ty = masses[tid]._y + masses[tid]._k2y * dt * 0.5f;
		masses[tid]._tz = masses[tid]._z + masses[tid]._k2z * dt * 0.5f;
	}
}

void SigAsiaDemo::MassList::evaluateK2(float dt)
{
	if (_computing) {
		std::cout << "Evaluate K2 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK2<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
	}
}

__global__ void deviceEvaluateK3(float dt, int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k3
		masses[tid]._k3x += ax * dt;
		masses[tid]._k3y += ay * dt;
		masses[tid]._k3z += az * dt;

		// set temporary position for k4
		masses[tid]._tx = masses[tid]._x + masses[tid]._k3x * dt;
		masses[tid]._ty = masses[tid]._y + masses[tid]._k3y * dt;
		masses[tid]._tz = masses[tid]._z + masses[tid]._k3z * dt;
	}
}

void SigAsiaDemo::MassList::evaluateK3(float dt)
{
	if (_computing) {
		std::cout << "Evaluate K3 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK3<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
	}
}

__global__ void deviceEvaluateK4(float dt, int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k4
		masses[tid]._k4x += ax * dt;
		masses[tid]._k4y += ay * dt;
		masses[tid]._k4z += az * dt;
	}
}

void SigAsiaDemo::MassList::evaluateK4(float dt)
{
	if (_computing) {
		std::cout << "Evaluate K4 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK4<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
	}
}

__global__ void deviceUpdate(float dt, int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < N) {
		masses[tid]._x += 0.166666666666667f * (
			masses[tid]._k1x +
			2.0f*masses[tid]._k2x +
			2.0f*masses[tid]._k3x +
			masses[tid]._k4x);
		masses[tid]._y += 0.166666666666667f * (
			masses[tid]._k1y +
			2.0f*masses[tid]._k2y +
			2.0f*masses[tid]._k3y +
			masses[tid]._k4y);
		masses[tid]._z += 0.166666666666667f * (
			masses[tid]._k1z +
			2.0f*masses[tid]._k2z +
			2.0f*masses[tid]._k3z +
			masses[tid]._k4z);
	}
}

void SigAsiaDemo::MassList::update(float dt)
{
	if (_computing) {
		std::cout << "Update masses (" << _masses.size() << ")." << std::endl;
		deviceUpdate<<<_masses.size(), 1>>>(dt, _masses.size(), _device_masses);
	}
}
