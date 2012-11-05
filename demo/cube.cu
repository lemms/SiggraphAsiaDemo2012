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
	size_t x, // multiple of 2
	size_t y,
	size_t z,
	float spacing,
	float mass,
	float radius) :
		_start(0),
		_end(0),
		_half_x(static_cast<int>(x/2)),
		_half_y(static_cast<int>(y/2)),
		_half_z(static_cast<int>(z/2)),
		_spacing(spacing),
		_mass(mass),
		_radius(radius)
{}

SigAsiaDemo::Cube::~Cube()
{} 

void SigAsiaDemo::Cube::create(
	float x,
	float y,
	float z,
	MassList &masses,
	SpringList &springs)
{
	_start = masses.size();
	std::cout << "Starting at index " << _start << "." << std::endl;

	int side = _half_x*2+1;
	int plane = side*side;
	//int cube = plane*side;

	// add points
	for (int i = -_half_x; i <= _half_x; ++i) {
		for (int j = -_half_y; j <= _half_y; ++j) {
			for (int k = -_half_z; k <= _half_z; ++k) {
				masses.push(SigAsiaDemo::Mass(
					_mass,
					static_cast<float>(i)*_spacing + x,
					static_cast<float>(j)*_spacing + y,
					static_cast<float>(k)*_spacing + z,
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
				if (right >= 0 && down >= 0) {
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
				if (back >= 0 && down >= 0) {
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
				if (back >= 0 && right >= 0) {
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
				}
			}
		}
	}
}
