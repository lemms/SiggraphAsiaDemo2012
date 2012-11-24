/*
Siggraph Asia 2012 Demo

Mass vector implementation.

Laurence Emms
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <iterator>
#include <cfloat>

#ifdef WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

// GLM
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mass.h"
#include "spring.h"
#include "creator.h"
#include "cube.h"

SigAsiaDemo::Cube::Cube(
	float x,
	float y,
	float z,
	size_t size_x, // multiple of 2
	size_t size_y,
	size_t size_z,
	float spacing,
	float mass,
	float radius) :
		_start(0),
		_end(0),
		_x(x),
		_y(y),
		_z(z),
		_half_x(static_cast<int>(size_x/2)),
		_half_y(static_cast<int>(size_y/2)),
		_half_z(static_cast<int>(size_z/2)),
		_spacing(spacing),
		_mass(mass),
		_radius(radius),
		_min_x(0.0),
		_min_y(0.0),
		_min_z(0.0),
		_max_x(0.0),
		_max_y(0.0),
		_max_z(0.0)
{}

SigAsiaDemo::Cube::~Cube()
{} 

void SigAsiaDemo::Cube::create(
	MassList &masses,
	SpringList &springs)
{
	_start = masses.size();
	std::cout << "Starting at index " << _start << "." << std::endl;

	int side = _half_x*2+1;
	int plane = side*side;

	// compute min/max
	_min_x = static_cast<float>(-_half_x)*_spacing + _x - _radius;
	_min_y = static_cast<float>(-_half_y)*_spacing + _y - _radius;
	_min_z = static_cast<float>(-_half_z)*_spacing + _z - _radius;
	_max_x = static_cast<float>(_half_x)*_spacing + _x + _radius;
	_max_y = static_cast<float>(_half_y)*_spacing + _y + _radius;
	_max_z = static_cast<float>(_half_z)*_spacing + _z + _radius;

	// add points
	for (int i = -_half_x; i <= _half_x; ++i) {
		for (int j = -_half_y; j <= _half_y; ++j) {
			for (int k = -_half_z; k <= _half_z; ++k) {
				masses.push(SigAsiaDemo::Mass(
					_mass,
					static_cast<float>(i)*_spacing + _x,
					static_cast<float>(j)*_spacing + _y,
					static_cast<float>(k)*_spacing + _z,
					0.0, 0.0, 0.0,
					0,
					_radius));
			}
		}
	}
	_end = masses.size();
	std::cout << "Ending at index " << _end << "." << std::endl;

	// add structural springs
	for (int i = -_half_x; i <= _half_x; ++i) {
		for (int j = -_half_y; j <= _half_y; ++j) {
			for (int k = -_half_z; k <= _half_z; ++k) {
				int ind_i = i + _half_x;
				int ind_j = j + _half_y;
				int ind_k = k + _half_z;

				int index = _start + ind_i + ind_j*side + ind_k*plane;
				// add springs to neighbors
				int right = -1;
				int down = -1;
				int back = -1;

				// compute indices
				if (i+1 <= _half_x)
					right = _start + (ind_i+1) + ind_j*side + ind_k*plane;
				if (j+1 <= _half_y)
					down = _start + ind_i + (ind_j+1)*side + ind_k*plane;
				if (k+1 <= _half_z)
					back = _start + ind_i + ind_j*side + (ind_k+1)*plane;

				// add springs
				if (right >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						right));
				}

				if (down >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						down));
				}

				if (back >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						back));
				}
			}
		}
	}

	// add bending springs
	for (int i = -_half_x; i <= _half_x; ++i) {
		for (int j = -_half_y; j <= _half_y; ++j) {
			for (int k = -_half_z; k <= _half_z; ++k) {
				int ind_i = i + _half_x;
				int ind_j = j + _half_y;
				int ind_k = k + _half_z;

				int index = _start + ind_i + ind_j*side + ind_k*plane;
				// add springs to neighbors
				int right = -1;
				int down = -1;
				int back = -1;

				// compute indices
				if (i+2 <= _half_x)
					right = _start + (ind_i+2) + ind_j*side + ind_k*plane;
				if (j+2 <= _half_y)
					down = _start + ind_i + (ind_j+2)*side + ind_k*plane;
				if (k+2 <= _half_z)
					back = _start + ind_i + ind_j*side + (ind_k+2)*plane;

				// add springs
				if (right >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						right));
				}

				if (down >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						down));
				}

				if (back >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						back));
				}
			}
		}
	}

	// add shear springs
	for (int i = -_half_x; i <= _half_x; ++i) {
		for (int j = -_half_y; j <= _half_y; ++j) {
			for (int k = -_half_z; k <= _half_z; ++k) {
				int ind_i = i + _half_x;
				int ind_j = j + _half_y;
				int ind_k = k + _half_z;

				// add springs to neighbors

				// front plane indices
				int index = _start + ind_i + ind_j*side + ind_k*plane;
				int right = -1;
				int down = -1;
				int right_down = -1;

				// back plane indices
				int back = -1;
				int back_right = -1;
				int back_down = -1;
				int back_right_down = -1;

				// compute indices
				if (i+1 <= _half_x) {
					right = _start +
					(ind_i+1) +
					ind_j*side +
					ind_k*plane;
				}
				if (j+1 <= _half_y) {
					down =
						_start +
						ind_i +
						(ind_j+1)*side +
						ind_k*plane;
				}
				if (i+1 <= _half_x && j+1 <= _half_y) {
					right_down =
						_start +
						(ind_i+1) +
						(ind_j+1)*side +
						ind_k*plane;
				}

				if (k+1 <= _half_z) {
					back = _start + ind_i + ind_j*side + (ind_k+1)*plane;
					if (i+1 <= _half_x) {
						back_right =
							_start +
							(ind_i+1) +
							ind_j*side +
							(ind_k+1)*plane;
					}
					if (j+1 <= _half_y) {
						back_down =
							_start +
							ind_i +
							(ind_j+1)*side +
							(ind_k+1)*plane;
					}
					if (i+1 <= _half_x && j+1 <= _half_y) {
						back_right_down = 
							_start +
							(ind_i+1) +
							(ind_j+1)*side +
							(ind_k+1)*plane;
					}

				}

				// add planar springs
				// front plane
				if (right_down >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						right_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						right,
						down));
				}

				// left plane
				if (back_down >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						back_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back,
						down));
				}

				// top plane
				if (back_right >= 0) {
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						back_right));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back,
						right));
				}

				if (back_right_down >= 0) {
					// back plane
					springs.push(SigAsiaDemo::Spring(
						masses,
						back,
						back_right_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back_right,
						back_down));

					// right plane
					springs.push(SigAsiaDemo::Spring(
						masses,
						right,
						back_right_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back_right,
						right_down));

					// bottom plane
					springs.push(SigAsiaDemo::Spring(
						masses,
						down,
						back_right_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back_down,
						right_down));

					// add core springs
					springs.push(SigAsiaDemo::Spring(
						masses,
						index,
						back_right_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back,
						right_down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						back_right,
						down));
					springs.push(SigAsiaDemo::Spring(
						masses,
						right,
						back_down));
				}
			}
		}
	}
}

SigAsiaDemo::CubeList::CubeList(
	float ks,
	float kd,
	unsigned int threads) :
		_ks(ks),
		_kd(kd),
		_threads(threads)
{}

void SigAsiaDemo::CubeList::setConstants(
	float ks,
	float kd)
{
	_ks = ks;
	_kd = kd;
}

void SigAsiaDemo::CubeList::push(
	Cube cube)
{
	_cubes.push_back(cube);
}

bool SigAsiaDemo::CubeList::empty() const
{
	return _cubes.empty();
}

size_t SigAsiaDemo::CubeList::size() const
{
	return _cubes.size();
}

void SigAsiaDemo::CubeList::create(
	MassList &masses,
	SpringList &springs)
{
	for (std::vector<Cube>::iterator i = _cubes.begin();
		i != _cubes.end(); i++) {
		i->create(masses, springs);
	}
}

SigAsiaDemo::Cube *SigAsiaDemo::CubeList::getCube(
	size_t index)
{
	if (index < _cubes.size()) {
		return &_cubes[index];
	}
	return NULL;
}

void SigAsiaDemo::CubeList::computeBounds(
	MassList &masses)
{
	// TODO: convert this to use parallel prefix min/max?
	masses.download();

	for (std::vector<Cube>::iterator cube = _cubes.begin();
		cube != _cubes.end(); cube++) {
		float min_x = FLT_MAX;
		float min_y = FLT_MAX;
		float min_z = FLT_MAX;
		float max_x = -FLT_MAX;
		float max_y = -FLT_MAX;
		float max_z = -FLT_MAX;
		for (unsigned int i = cube->_start; i < cube->_end; i++) {
			Mass *mass = masses.getMass(i);
			if (mass->_x - mass->_radius < min_x)
				min_x = mass->_x - mass->_radius;
			if (mass->_y - mass->_radius < min_y)
				min_y = mass->_y - mass->_radius;
			if (mass->_z - mass->_radius < min_z)
				min_z = mass->_z - mass->_radius;

			if (mass->_x + mass->_radius > max_x)
				max_x = mass->_x + mass->_radius;
			if (mass->_y + mass->_radius > max_y)
				max_y = mass->_y + mass->_radius;
			if (mass->_z + mass->_radius > max_z)
				max_z = mass->_z + mass->_radius;
		}

		cube->_min_x = min_x;
		cube->_min_y = min_y;
		cube->_min_z = min_z;

		cube->_max_x = max_x;
		cube->_max_y = max_y;
		cube->_max_z = max_z;

		/*
		std::cout << "Start: " << cube->_start << std::endl;
		std::cout << "End: " << cube->_end << std::endl;
		std::cout << "[" << cube->_min_x << ", " \
		<< cube->_max_x << "]" << std::endl;
		std::cout << "[" << cube->_min_y << ", " \
		<< cube->_max_y << "]" << std::endl;
		std::cout << "[" << cube->_min_z << ", " \
		<< cube->_max_z << "]" << std::endl;
		*/
	}
}

__global__ void deviceCollideCubes(
	float dt,
	float ks,
	float kd,
	unsigned int masses_count,
	unsigned int collider_start,
	unsigned int collidee_start,
	unsigned int collidee_end,
	unsigned int masses_size,
	SigAsiaDemo::Mass *masses)
{
	// compute collision forces
	// as temporary springs
	// O(N^2) comparison on the GPU

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < masses_count) {
		for (unsigned int i = collidee_start; i < collidee_end; i++) {
			float l0 = masses[collider_start+tid]._radius + masses[i]._radius;
			
			// d contains the vector from mass 0 to mass 1
			float dx = masses[i]._x - masses[collider_start+tid]._x;
			float dy = masses[i]._y - masses[collider_start+tid]._y;
			float dz = masses[i]._z - masses[collider_start+tid]._z;

			// compute length of d
			float ld = sqrt(dx*dx + dy*dy + dz*dz);

			if (ld < l0) {
				// velocity delta
				float dvx =
					masses[i]._vx - masses[collider_start+tid]._vx;
				float dvy =
					masses[i]._vy - masses[collider_start+tid]._vy;
				float dvz =
					masses[i]._vz - masses[collider_start+tid]._vz;

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
				// compute impulse for mass 1
				float impulse = (
					-ks * (ld / l0 - 1.0f)
					-kd * dot_dv_v) * dt;
				float i_x = impulse * udx;
				float i_y = impulse * udy;
				float i_z = impulse * udz;

				// compute impulse for mass 0
				masses[collider_start+tid]._vx += -i_x;
				masses[collider_start+tid]._vy += -i_y;
				masses[collider_start+tid]._vz += -i_z;
				masses[collider_start+tid]._tvx += -i_x;
				masses[collider_start+tid]._tvy += -i_y;
				masses[collider_start+tid]._tvz += -i_z;
			}
		}
	}
}

void SigAsiaDemo::CubeList::collideCubes(
	float dt,
	MassList &masses)
{
	// get overlapping pairs of cubes
	for (unsigned int i = 0; i < _cubes.size(); ++i) {
		for (unsigned int j = i+1; j < _cubes.size(); ++j) {
			if (
				_cubes[i]._min_x > _cubes[j]._max_x ||
				_cubes[j]._min_x > _cubes[i]._max_x ||
				_cubes[i]._min_y > _cubes[j]._max_y ||
				_cubes[j]._min_y > _cubes[i]._max_y ||
				_cubes[i]._min_z > _cubes[j]._max_z ||
				_cubes[j]._min_z > _cubes[i]._max_z) {
				// no overlap

				/*
				std::cout << "Cube " << i << " does not overlap " \
				<< j << std::endl;
				std::cout << "Cube " << i << ":" << std::endl;
				std::cout << "[" << _cubes[i]._min_x << ", " \
				<< _cubes[i]._max_x << "]" << std::endl;
				std::cout << "[" << _cubes[i]._min_y << ", " \
				<< _cubes[i]._max_y << "]" << std::endl;
				std::cout << "[" << _cubes[i]._min_z << ", " \
				<< _cubes[i]._max_z << "]" << std::endl;
				std::cout << "Cube " << j << ":" << std::endl;
				std::cout << "[" << _cubes[j]._min_x << ", " \
				<< _cubes[j]._max_x << "]" << std::endl;
				std::cout << "[" << _cubes[j]._min_y << ", " \
				<< _cubes[j]._max_y << "]" << std::endl;
				std::cout << "[" << _cubes[j]._min_z << ", " \
				<< _cubes[j]._max_z << "]" << std::endl;
				*/
			} else {
				// overlap

				/*
				std::cout << "Cube " << i << " overlaps " << j << std::endl;
				std::cout << "Cube " << i << ":" << std::endl;
				std::cout << "[" << _cubes[i]._min_x << ", " \
				<< _cubes[i]._max_x << "]" << std::endl;
				std::cout << "[" << _cubes[i]._min_y << ", " \
				<< _cubes[i]._max_y << "]" << std::endl;
				std::cout << "[" << _cubes[i]._min_z << ", " \
				<< _cubes[i]._max_z << "]" << std::endl;
				std::cout << "Cube " << j << ":" << std::endl;
				std::cout << "[" << _cubes[j]._min_x << ", " \
				<< _cubes[j]._max_x << "]" << std::endl;
				std::cout << "[" << _cubes[j]._min_y << ", " \
				<< _cubes[j]._max_y << "]" << std::endl;
				std::cout << "[" << _cubes[j]._min_z << ", " \
				<< _cubes[j]._max_z << "]" << std::endl;
				*/

				unsigned int masses_count = _cubes[i]._end - _cubes[i]._start;

				deviceCollideCubes
					<<<(masses_count+_threads-1)/_threads, _threads>>>(
						dt,
						_ks,
						_kd,
						masses_count,
						_cubes[i]._start,
						_cubes[j]._start,
						_cubes[j]._end,
						masses.size(),
						masses.getDeviceMasses());
				cudaThreadSynchronize();

				masses_count = _cubes[j]._end - _cubes[j]._start;

				deviceCollideCubes
					<<<(masses_count+_threads-1)/_threads, _threads>>>(
						dt,
						_ks,
						_kd,
						masses_count,
						_cubes[j]._start,
						_cubes[i]._start,
						_cubes[i]._end,
						masses.size(),
						masses.getDeviceMasses());

				cudaThreadSynchronize();

			}
		}
	}
}
