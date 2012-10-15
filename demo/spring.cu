/*
Siggraph Asia 2012 Demo

Spring vector implementation

Laurence Emms
*/

#include <iostream>
#include <vector>
#include <cmath>
#include "mass.h"
#include "spring.h"

SigAsiaDemo::Spring::Spring(
	MassList &masses,
	unsigned int mass0,
	unsigned int mass1,
	float ks,
	float kd) :
		_mass0(mass0),
		_mass1(mass1),
		_ks(ks),
		_kd(kd),
		_l0(0.0),
		_fx0(0.0),
		_fy0(0.0),
		_fz0(0.0),
		_fx1(0.0),
		_fy1(0.0),
		_fz1(0.0)
{
	// compute l0
	Mass *m0 = masses.getMass(_mass0);
	Mass *m1 = masses.getMass(_mass1);
	if (!m0) {
		std::cerr << "Spring pointing to null mass 0" << std::endl;
		std::terminate();
	}
	if (!m1) {
		std::cerr << "Spring pointing to null mass 1" << std::endl;
		std::terminate();
	}
	float dx = m0->_x - m1->_x;
	float dy = m0->_y - m1->_y;
	float dz = m0->_z - m1->_z;
	_l0 = sqrt(dx*dx + dy*dy + dz*dz);
}

SigAsiaDemo::SpringList::SpringList() :
	_computing(false),
	_changed(false),
	_device_springs(0)
{
}

SigAsiaDemo::SpringList::~SpringList()
{
	if (_device_springs) {
		cudaFree(_device_springs);
	}
}

bool SigAsiaDemo::SpringList::push(Spring spring)
{
	// enforce that no springs can be added
	// if the buffer is uploaded to the GPU
	if (!_computing) {
		_springs.push_back(spring);
		_changed = true;
		return true;
	}
	return false;
}

size_t SigAsiaDemo::SpringList::size() const
{
	return _springs.size();
}

void SigAsiaDemo::SpringList::upload()
{
	if (_changed) {
		std::cout << "Upload springs" << std::endl;
		_changed = false;
		if (_device_springs) {
			std::cout << "Free springs." << std::endl;
			cudaFree(_device_springs);
		}

		// allocate GPU buffer
		std::cout << "Allocate GPU buffer of size " << \
		_springs.size() << "." << std::endl;
		cudaMalloc(
			(void**)&_device_springs,
			_springs.size()*sizeof(Spring));

		// copy into GPU buffer
		std::cout << "Copy springs into GPU buffer." << std::endl;
		cudaMemcpy(
			_device_springs,
			&_springs[0],
			_springs.size()*sizeof(Spring),
			cudaMemcpyHostToDevice);
	}

	_computing = true;
}

void SigAsiaDemo::SpringList::download()
{
	if (_changed) {
		std::cerr << "Error: Spring list changed while \
data was being used in GPU computations." << std::endl;
		std::terminate();
	} else {
		std::cout << "Download springs" << std::endl;
		// copy into CPU buffer
		std::cout << "Copy springs into CPU buffer." << std::endl;
		cudaMemcpy(
			&_springs[0],
			_device_springs,
			_springs.size()*sizeof(Spring),
			cudaMemcpyDeviceToHost);
	}
	_computing = false;
}

SigAsiaDemo::Spring *SigAsiaDemo::SpringList::getSpring(size_t index)
{
	if (_springs.empty()) {
		std::cout << "Warning: getSpring called on \
empty spring list." << std::endl;
		return 0;
	}
	if (_computing) {
		std::cout << "Warning: getSpring called while \
spring list is uploaded to the GPU." << std::endl;
		return 0;
	}
	if (index >= _springs.size()) {
		std::cout << "Warning: getSpring called on index \
out of bounds." << std::endl;
		return 0;
	}

	return &_springs[index];
}

SigAsiaDemo::Spring *SigAsiaDemo::SpringList::getDeviceSprings()
{
	return _device_springs;
}

__global__ void deviceComputeSpringForces(
	unsigned int springs_size,
	SigAsiaDemo::Spring *springs,
	unsigned int masses_size,
	SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
	if (tid < springs_size) {
		// v is the vector from mass 1 to mass 0
		// we're operating on the temporary position
		float vx =
			masses[springs[tid]._mass0]._tx - masses[springs[tid]._mass1]._tx;
		float vy =
			masses[springs[tid]._mass0]._ty - masses[springs[tid]._mass1]._ty;
		float vz =
			masses[springs[tid]._mass0]._tz - masses[springs[tid]._mass1]._tz;
		// compute length of v
		float lv = sqrt(vx*vx + vy*vy + vz*vz);
		float rcp_lv = 1.0f / lv;
		// compute unit v
		float uvx = vx * rcp_lv;
		float uvy = vy * rcp_lv;
		float uvz = vz * rcp_lv;
		// project temporary velocity of mass 0 onto v
		float dot_tv0_v =
			masses[springs[tid]._mass0]._tvx * uvx +
			masses[springs[tid]._mass0]._tvy * uvy +
			masses[springs[tid]._mass0]._tvz * uvz;
		float tv0x = uvx * dot_tv0_v;
		float tv0y = uvy * dot_tv0_v;
		float tv0z = uvz * dot_tv0_v;
		// compute force for mass 0 to mass 1
		float extension = -springs[tid]._ks * (lv / springs[tid]._l0 - 1.0f);
		springs[tid]._fx0 = extension * uvx - springs[tid]._kd * tv0x;
		springs[tid]._fy0 = extension * uvy - springs[tid]._kd * tv0y;
		springs[tid]._fz0 = extension * uvz - springs[tid]._kd * tv0z;
	}
}

void SigAsiaDemo::SpringList::applySpringForces(MassList &masses)
{
	if (_computing) {
		std::cout << "Compute spring forces (" << _springs.size() << ")." \
		<< std::endl;
		deviceComputeSpringForces<<<_springs.size(), 1>>>(
			_springs.size(),
			_device_springs,
			masses.size(),
			masses.getDeviceMasses());
		std::cout << "Accumulate mass forces (" << masses.size() << ")." \
		<< std::endl;
		// TODO
	}
}
