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
		_size(0),
		_half_x(static_cast<int>(x/2)),
		_half_y(static_cast<int>(y/2)),
		_half_z(static_cast<int>(z/2)),
		_spacing(spacing),
		_mass(mass),
		_radius(radius)
{}

SigAsiaDemo::Cube::~Cube()
{
}

void SigAsiaDemo::Cube::create(
	float x,
	float y,
	float z,
	MassList &masses,
	SpringList &springs)
{
	// add points

	for (int i = -_half_x; i < _half_x; ++i) {
		for (int j = -_half_y; j < _half_y; ++j) {
			for (int k = -_half_z; k < _half_z; ++k) {
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
}
