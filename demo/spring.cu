/*
Siggraph Asia 2012 Demo

Spring vector implementation.

Laurence Emms
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
using namespace std;

#ifdef WIN32
#include <windows.h>
#define GLEW_STATIC
#endif
#include <GL/glew.h>

// GLM
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mass.h"
#include "spring.h"
#include "creator.h"
#include "cube.h"

SigAsiaDemo::Spring::Spring(
	MassList &masses,
	unsigned int mass0,
	unsigned int mass1) :
		_mass0(mass0),
		_mass1(mass1),
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
		cerr << "Spring pointing to null mass 0" << endl;
		terminate();
	}
	if (!m1) {
		cerr << "Spring pointing to null mass 1" << endl;
		terminate();
	}
	float dx = m0->_x - m1->_x;
	float dy = m0->_y - m1->_y;
	float dz = m0->_z - m1->_z;
	_l0 = sqrt(dx*dx + dy*dy + dz*dz);
}

SigAsiaDemo::SpringList::SpringList(
	float ks,
	float kd,
	unsigned int threads) :
	_ks(ks),
	_kd(kd),
	_computing(false),
	_changed(false),
	_device_springs(0),
	_device_mass_spring_counts(0),
	_device_mass_spring_indices(0),
	_threads(threads)
{}

SigAsiaDemo::SpringList::~SpringList()
{
	if (_computing) {
		cerr << "Warning: Still computing!" << endl;
	}
	if (_device_springs) {
		// free springs
		cudaFree(_device_springs);
		_device_springs = 0;
	}

	if (_device_mass_spring_counts) {
		// free counts
		cudaFree(_device_mass_spring_counts);
		_device_mass_spring_counts = 0;
	}

	if (_device_mass_spring_indices) {
		// free indices
		cudaFree(_device_mass_spring_indices);
		_device_mass_spring_indices = 0;
	}
}

void SigAsiaDemo::SpringList::setConstants(
	float ks,
	float kd)
{
	_ks = ks;
	_kd = kd;
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

bool SigAsiaDemo::SpringList::empty() const
{
	return _springs.empty();
}

size_t SigAsiaDemo::SpringList::size() const
{
	return _springs.size();
}

// Note: this must be called before updating the mass list
void SigAsiaDemo::SpringList::upload(MassList &masses, CubeList &cubes)
{
	if (_computing) {
		// do nothing if computing
		return;
	}

	// if masses have been changed or springs have been changed, the mapping
	// between masses and springs will need to be recomputed
	
	// NOTE: we assume that the mass list has not changed, otherwise the spring
	// list is invalid and needs to be changed as well
	if (_changed) {
		// Update mass -> spring mapping.
		// NOTE: hopefully this won't change too often, otherwise a GPU 
		// solution will have to be developed for updating this mapping

		// clear counts and indices
		_mass_spring_counts.clear();
		_mass_spring_indices.clear();
		size_t count = 0;
		size_t counts_index = 1;
		size_t indices_index = 0;
		// contains the number of spring indices for each mass
		// the last element of this array is the total indices
		// array size
		_mass_spring_counts.resize(masses.size() + 1);
		_mass_spring_indices.resize(_springs.size() * 2);
		_mass_spring_counts[0] = 0;

		for (size_t c = 0; c < cubes.size(); ++c) {
			Cube *cube = cubes.getCube(c);
			cout << "cube: " << c << endl;
			cout << "masses: " << cube->_start << ", " << cube->_end << endl;
			cout << "springs: " << cube->_spring_start << ", " << cube->_spring_end << endl;
			for (size_t i = cube->_start; i < cube->_end; ++i) {
				const Mass *m = masses.getMass(i);
				if (!m) {
					cerr << "Error: Failed to get mass " << i << endl;
					terminate();
				}
				for (size_t j = cube->_spring_start; j < cube->_spring_end; ++j) {
					if (_springs[j]._mass0 == i || _springs[j]._mass1 == i) {
						if (_springs[j]._mass0 == i && _springs[j]._mass1 == i) {
							cerr << \
							"Error: Both spring indices point to the same mass." \
							<< endl;
							terminate();
						}

						// increment counts for this mass
						if (indices_index >= _mass_spring_indices.size()) {
							cerr << "Error: indices_index exceeds expected \
	indices size." << endl;
							terminate();
						}
						_mass_spring_indices[indices_index] = j;
						indices_index++;
						count++;
					}
				}
				if (counts_index >= _mass_spring_counts.size()) {
					cerr << "Error: counts_index exceeds expected \
	counts size." << endl;
					terminate();
				}
				_mass_spring_counts[counts_index] = count;
				counts_index++;
			}
		}
		
		// allocate GPU buffers
		cudaError_t result = cudaMalloc(
			(void**)&_device_mass_spring_counts,
			_mass_spring_counts.size()*sizeof(unsigned int));
		if (result != cudaSuccess) {
			cerr << "Error: CUDA failed to malloc memory." << endl;
			terminate();
		}

		result = cudaMalloc(
			(void**)&_device_mass_spring_indices,
			_mass_spring_indices.size()*sizeof(unsigned int));
		if (result != cudaSuccess) {
			cerr << "Error: CUDA failed to malloc memory." << endl;
			terminate();
		}

		// copy into GPU buffer
		cudaMemcpy(
			_device_mass_spring_counts,
			&_mass_spring_counts[0],
			_mass_spring_counts.size()*sizeof(unsigned int),
			cudaMemcpyHostToDevice);

		cudaMemcpy(
			_device_mass_spring_indices,
			&_mass_spring_indices[0],
			_mass_spring_indices.size()*sizeof(unsigned int),
			cudaMemcpyHostToDevice);

		// upload springs
		_changed = false;
		if (_device_springs) {
			// free springs
			cudaFree(_device_springs);
			_device_springs = 0;
		}

		// allocate GPU buffer
		result = cudaMalloc(
			(void**)&_device_springs,
			_springs.size()*sizeof(Spring));
		if (result != cudaSuccess) {
			cerr << "Error: CUDA failed to malloc memory." << endl;
			terminate();
		}

		// copy into GPU buffer
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
		cerr << "Error: Spring list changed while \
data was being used in GPU computations." << endl;
		terminate();
	} else {
		// copy into CPU buffer
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
		cerr << "getSpring called on \
empty spring list." << endl;
		return 0;
	}
	if (index >= _springs.size()) {
		cerr << "Warning: getSpring called on index \
out of bounds." << endl;
		return 0;
	}

	return &_springs[index];
}

SigAsiaDemo::Spring *SigAsiaDemo::SpringList::getDeviceSprings()
{
	return _device_springs;
}

bool SigAsiaDemo::SpringList::getChanged() const
{
	return _changed;
}

__global__ void deviceComputeSpringForces(
	float ks,
	float kd,
	unsigned int springs_size,
	SigAsiaDemo::Spring *springs,
	unsigned int masses_size,
	SigAsiaDemo::Mass *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < springs_size) {
		// d is the vector from mass 0 to mass 1
		// we're operating on the temporary position

		// position delta
		float dx =
			masses[springs[tid]._mass1]._tx - masses[springs[tid]._mass0]._tx;
		float dy =
			masses[springs[tid]._mass1]._ty - masses[springs[tid]._mass0]._ty;
		float dz =
			masses[springs[tid]._mass1]._tz - masses[springs[tid]._mass0]._tz;

		// velocity delta
		float dvx =
			masses[springs[tid]._mass1]._tvx -
			masses[springs[tid]._mass0]._tvx;
		float dvy =
			masses[springs[tid]._mass1]._tvy -
			masses[springs[tid]._mass0]._tvy;
		float dvz =
			masses[springs[tid]._mass1]._tvz -
			masses[springs[tid]._mass0]._tvz;

		// compute length of d
		float ld = sqrt(dx*dx + dy*dy + dz*dz);
		float rcp_ld = 1.0f;
		if (ld != 0.0f) {
			rcp_ld = 1.0f / ld;
		}
		// compute unit d
		float udx = dx * rcp_ld;
		float udy = dy * rcp_ld;
		float udz = dz * rcp_ld;

		// project velocity delta onto unit d
		float dot_dv_v =
			dvx * udx +
			dvy * udy +
			dvz * udz;
		// compute force for mass 1
		float force =
			-ks * (ld / springs[tid]._l0 - 1.0f)
			-kd * dot_dv_v;
		springs[tid]._fx1 = force * udx;
		springs[tid]._fy1 = force * udy;
		springs[tid]._fz1 = force * udz;

		// compute force for mass 0
		springs[tid]._fx0 = -springs[tid]._fx1;
		springs[tid]._fy0 = -springs[tid]._fy1;
		springs[tid]._fz0 = -springs[tid]._fz1;
	}
}

__global__ void deviceApplySpringForces(
	unsigned int springs_size,
	SigAsiaDemo::Spring *springs,
	unsigned int masses_size,
	SigAsiaDemo::Mass *masses,
	unsigned int *mass_spring_counts,
	unsigned int *mass_spring_indices)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < masses_size) {
		for (unsigned int i = mass_spring_counts[tid];
			i < mass_spring_counts[tid+1];
			++i) {
			unsigned int s = mass_spring_indices[i];
			if (tid == springs[s]._mass0) {
				masses[tid]._fx += springs[s]._fx0;
				masses[tid]._fy += springs[s]._fy0;
				masses[tid]._fz += springs[s]._fz0;
			} else if (tid == springs[s]._mass1) {
				masses[tid]._fx += springs[s]._fx1;
				masses[tid]._fy += springs[s]._fy1;
				masses[tid]._fz += springs[s]._fz1;
			}
		}
	}
}

void SigAsiaDemo::SpringList::applySpringForces(MassList &masses)
{
	if (_computing && !_springs.empty() && !masses.empty()) {
		deviceComputeSpringForces
			<<<(_springs.size()+_threads-1)/_threads, _threads>>>(
				_ks,
				_kd,
				_springs.size(),
				_device_springs,
				masses.size(),
				masses.getDeviceMasses());
		cudaThreadSynchronize();

		deviceApplySpringForces
			<<<(masses.size()+_threads-1)/_threads, _threads>>>(
				_springs.size(),
				_device_springs,
				masses.size(),
				masses.getDeviceMasses(),
				_device_mass_spring_counts,
				_device_mass_spring_indices);
		cudaThreadSynchronize();
	}
}
