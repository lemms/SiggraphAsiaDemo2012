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
#include <cmath>
#include <cfloat>

#ifdef WIN32
#include <windows.h>
#define GLEW_STATIC
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
		_size_x(size_x),
		_size_y(size_y),
		_size_z(size_z),
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

	int side = _size_x+1;
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
	size_t res_x,
	size_t res_y,
	size_t res_z,
	float ks,
	float kd,
	unsigned int threads) :
		_res_x(res_x),
		_res_y(res_y),
		_res_z(res_z),
		_ks(ks),
		_kd(kd),
		_threads(threads),
		_cube_ModelViewLocation(0),
		_cube_ProjectionLocation(0),
		_cube_NormalLocation(0),
		_cube_vertex_shader(0),
		_cube_fragment_shader(0),
		_cube_program(0),
		_cube_array(0),
		_cube_pos_buffer(0),
		_cube_norm_buffer(0)
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

void compute_triangle(
	float threshold,
	size_t ind_0,
	float x0, float y0, float z0,
	float nx0, float ny0, float nz0,
	size_t ind_1,
	float x1, float y1, float z1,
	float nx1, float ny1, float nz1,
	size_t ind_2,
	float x2, float y2, float z2,
	float nx2, float ny2, float nz2,
	size_t ind_3,
	float x3, float y3, float z3,
	float nx3, float ny3, float nz3,
	std::vector<float> &weights,
	std::vector<float> &tri_positions,
	std::vector<float> &tri_normals)
{
	if (	weights[ind_0] < threshold &&
		weights[ind_1] >= threshold &&
		weights[ind_2] >= threshold &&
		weights[ind_3] >= threshold) {
		tri_positions.push_back((x0+x1)*0.5);
		tri_positions.push_back((y0+y1)*0.5);
		tri_positions.push_back((z0+z1)*0.5);
		tri_normals.push_back((nx0+nx1)*0.5);
		tri_normals.push_back((ny0+ny1)*0.5);
		tri_normals.push_back((nz0+nz1)*0.5);

		tri_positions.push_back((x0+x2)*0.5);
		tri_positions.push_back((y0+y2)*0.5);
		tri_positions.push_back((z0+z2)*0.5);
		tri_normals.push_back((nx0+nx2)*0.5);
		tri_normals.push_back((ny0+ny2)*0.5);
		tri_normals.push_back((nz0+nz2)*0.5);

		tri_positions.push_back((x0+x3)*0.5);
		tri_positions.push_back((y0+y3)*0.5);
		tri_positions.push_back((z0+z3)*0.5);
		tri_normals.push_back((nx0+nx3)*0.5);
		tri_normals.push_back((ny0+ny3)*0.5);
		tri_normals.push_back((nz0+nz3)*0.5);
	}
	if (	weights[ind_0] >= threshold &&
		weights[ind_1] < threshold &&
		weights[ind_2] < threshold &&
		weights[ind_3] < threshold) {
		tri_positions.push_back((x0+x1)*0.5);
		tri_positions.push_back((y0+y1)*0.5);
		tri_positions.push_back((z0+z1)*0.5);
		tri_normals.push_back((nx0+nx1)*0.5);
		tri_normals.push_back((ny0+ny1)*0.5);
		tri_normals.push_back((nz0+nz1)*0.5);

		tri_positions.push_back((x0+x3)*0.5);
		tri_positions.push_back((y0+y3)*0.5);
		tri_positions.push_back((z0+z3)*0.5);
		tri_normals.push_back((nx0+nx3)*0.5);
		tri_normals.push_back((ny0+ny3)*0.5);
		tri_normals.push_back((nz0+nz3)*0.5);

		tri_positions.push_back((x0+x2)*0.5);
		tri_positions.push_back((y0+y2)*0.5);
		tri_positions.push_back((z0+z2)*0.5);
		tri_normals.push_back((nx0+nx2)*0.5);
		tri_normals.push_back((ny0+ny2)*0.5);
		tri_normals.push_back((nz0+nz2)*0.5);
	}
}

void compute_quad(
	float threshold,
	size_t ind_0,
	float x0, float y0, float z0,
	float nx0, float ny0, float nz0,
	size_t ind_1,
	float x1, float y1, float z1,
	float nx1, float ny1, float nz1,
	size_t ind_2,
	float x2, float y2, float z2,
	float nx2, float ny2, float nz2,
	size_t ind_3,
	float x3, float y3, float z3,
	float nx3, float ny3, float nz3,
	std::vector<float> &weights,
	std::vector<float> &tri_positions,
	std::vector<float> &tri_normals)
{
	if (	weights[ind_0] < threshold &&
		weights[ind_1] < threshold &&
		weights[ind_2] >= threshold &&
		weights[ind_3] >= threshold) {
		tri_positions.push_back((x0+x3)*0.5);
		tri_positions.push_back((y0+y3)*0.5);
		tri_positions.push_back((z0+z3)*0.5);
		tri_normals.push_back((nx0+nx3)*0.5);
		tri_normals.push_back((ny0+ny3)*0.5);
		tri_normals.push_back((nz0+nz3)*0.5);

		tri_positions.push_back((x1+x3)*0.5);
		tri_positions.push_back((y1+y3)*0.5);
		tri_positions.push_back((z1+z3)*0.5);
		tri_normals.push_back((nx1+nx3)*0.5);
		tri_normals.push_back((ny1+ny3)*0.5);
		tri_normals.push_back((nz1+nz3)*0.5);

		tri_positions.push_back((x0+x2)*0.5);
		tri_positions.push_back((y0+y2)*0.5);
		tri_positions.push_back((z0+z2)*0.5);
		tri_normals.push_back((nx0+nx2)*0.5);
		tri_normals.push_back((ny0+ny2)*0.5);
		tri_normals.push_back((nz0+nz2)*0.5);

		tri_positions.push_back((x0+x2)*0.5);
		tri_positions.push_back((y0+y2)*0.5);
		tri_positions.push_back((z0+z2)*0.5);
		tri_normals.push_back((nx0+nx2)*0.5);
		tri_normals.push_back((ny0+ny2)*0.5);
		tri_normals.push_back((nz0+nz2)*0.5);

		tri_positions.push_back((x1+x3)*0.5);
		tri_positions.push_back((y1+y3)*0.5);
		tri_positions.push_back((z1+z3)*0.5);
		tri_normals.push_back((nx1+nx3)*0.5);
		tri_normals.push_back((ny1+ny3)*0.5);
		tri_normals.push_back((nz1+nz3)*0.5);

		tri_positions.push_back((x1+x2)*0.5);
		tri_positions.push_back((y1+y2)*0.5);
		tri_positions.push_back((z1+z2)*0.5);
		tri_normals.push_back((nx1+nx2)*0.5);
		tri_normals.push_back((ny1+ny2)*0.5);
		tri_normals.push_back((nz1+nz2)*0.5);
	}
	if (	weights[ind_0] >= threshold &&
		weights[ind_1] < threshold &&
		weights[ind_2] < threshold &&
		weights[ind_3] < threshold) {
		tri_positions.push_back((x0+x3)*0.5);
		tri_positions.push_back((y0+y3)*0.5);
		tri_positions.push_back((z0+z3)*0.5);
		tri_normals.push_back((nx0+nx3)*0.5);
		tri_normals.push_back((ny0+ny3)*0.5);
		tri_normals.push_back((nz0+nz3)*0.5);

		tri_positions.push_back((x0+x2)*0.5);
		tri_positions.push_back((y0+y2)*0.5);
		tri_positions.push_back((z0+z2)*0.5);
		tri_normals.push_back((nx0+nx2)*0.5);
		tri_normals.push_back((ny0+ny2)*0.5);
		tri_normals.push_back((nz0+nz2)*0.5);

		tri_positions.push_back((x1+x3)*0.5);
		tri_positions.push_back((y1+y3)*0.5);
		tri_positions.push_back((z1+z3)*0.5);
		tri_normals.push_back((nx1+nx3)*0.5);
		tri_normals.push_back((ny1+ny3)*0.5);
		tri_normals.push_back((nz1+nz3)*0.5);

		tri_positions.push_back((x0+x2)*0.5);
		tri_positions.push_back((y0+y2)*0.5);
		tri_positions.push_back((z0+z2)*0.5);
		tri_normals.push_back((nx0+nx2)*0.5);
		tri_normals.push_back((ny0+ny2)*0.5);
		tri_normals.push_back((nz0+nz2)*0.5);

		tri_positions.push_back((x1+x2)*0.5);
		tri_positions.push_back((y1+y2)*0.5);
		tri_positions.push_back((z1+z2)*0.5);
		tri_normals.push_back((nx1+nx2)*0.5);
		tri_normals.push_back((ny1+ny2)*0.5);
		tri_normals.push_back((nz1+nz2)*0.5);

		tri_positions.push_back((x1+x3)*0.5);
		tri_positions.push_back((y1+y3)*0.5);
		tri_positions.push_back((z1+z3)*0.5);
		tri_normals.push_back((nx1+nx3)*0.5);
		tri_normals.push_back((ny1+ny3)*0.5);
		tri_normals.push_back((nz1+nz3)*0.5);
	}
}

void compute_triangles(
	float threshold,
	size_t ind_0,
	float x0, float y0, float z0,
	float nx0, float ny0, float nz0,
	size_t ind_1,
	float x1, float y1, float z1,
	float nx1, float ny1, float nz1,
	size_t ind_2,
	float x2, float y2, float z2,
	float nx2, float ny2, float nz2,
	size_t ind_3,
	float x3, float y3, float z3,
	float nx3, float ny3, float nz3,
	std::vector<float> &weights,
	std::vector<float> &tri_positions,
	std::vector<float> &tri_normals)
{
	// if all weights in this tetrahedron are > threshold or < threshold, no triangle is drawn

	// triangle conditions
	// index 0
	compute_triangle(threshold,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		weights, tri_positions, tri_normals);
	// index 1
	compute_triangle(threshold,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		weights, tri_positions, tri_normals);
	// index 2
	compute_triangle(threshold,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		weights, tri_positions, tri_normals);
	// index 3
	compute_triangle(threshold,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		weights, tri_positions, tri_normals);

	// index 3
	compute_triangle(threshold,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		weights, tri_positions, tri_normals);

	// index 01
	compute_quad(threshold,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		weights, tri_positions, tri_normals);

	// index 12
	compute_quad(threshold,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		weights, tri_positions, tri_normals);

	// index 02
	compute_quad(threshold,
		ind_0, x0, y0, z0, nx0, ny0, nz0,
		ind_2, x2, y2, z2, nx2, ny2, nz2,
		ind_1, x1, y1, z1, nx1, ny1, nz1,
		ind_3, x3, y3, z3, nx3, ny3, nz3,
		weights, tri_positions, tri_normals);
}

bool SigAsiaDemo::CubeList::loadShaders()
{
	std::cout << "Load cube shader" << std::endl;
	bool success = false;
	success = loadShader(
		"cubeVS.glsl",
		"",
		"cubeFS.glsl",
		&_cube_program,
		&_cube_vertex_shader,
		0,
		&_cube_fragment_shader);
	if (!success)
		return false;

	glUseProgram(_cube_program);

	// get uniforms
	_cube_ModelViewLocation = glGetUniformLocation(
		_cube_program, "ModelView");
	if (_cube_ModelViewLocation == -1) {
		std::cerr << "Error: Failed to get ModelView location." \
			<< std::endl;
		return false;
	}

	_cube_ProjectionLocation = glGetUniformLocation(
		_cube_program, "Projection");
	if (_cube_ProjectionLocation == -1) {
		std::cerr << "Error: Failed to get Projection location." \
			<< std::endl;
		return false;
	}

	_cube_NormalLocation = glGetUniformLocation(
		_cube_program, "Normal");
	if (_cube_NormalLocation == -1) {
		std::cerr << "Error: Failed to get Normal location." \
			<< std::endl;
		return false;
	}

	return true;
}

void SigAsiaDemo::CubeList::computeBounds(
	MassList &masses)
{
	// TODO: convert this to use parallel prefix min/max?
	masses.download();

	_tri_positions.clear();
	_tri_normals.clear();
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

		float range_x = max_x - min_x;
		float range_y = max_y - min_y;
		float range_z = max_z - min_z;
		
		float res_xf = static_cast<float>(_res_x);
		float res_yf = static_cast<float>(_res_y);
		float res_zf = static_cast<float>(_res_z);

		float ux = res_xf / range_x;
		float uy = res_yf / range_y;
		float uz = res_zf / range_z;
		float delta_x = 1.0 / ux;
		float delta_y = 1.0 / uy;
		float delta_z = 1.0 / uz;

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

		// generate surface mesh

		//std::cout << "Fill _tri_positions & _tri_normals" << std::endl;

		size_t x_size = cube->_size_x+1;
		size_t y_size = cube->_size_y+1;
		size_t z_size = cube->_size_z+1;
		size_t xy_size = x_size * y_size;
		//size_t index = cube->_start + i + j * x_size + k * xy_size;

		// front and back face
		for (size_t i = 0; i < x_size-1; ++i) {
			for (size_t j = 0; j < y_size-1; ++j) {
				// front face
				{
					size_t i00 = cube->_start + i + j * x_size;
					size_t i10 = cube->_start + (i+1) + j * x_size;
					size_t i01 = cube->_start + i + (j+1) * x_size;
					size_t i11 = cube->_start + (i+1) + (j+1) * x_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					float x1000 = mass10->_x - mass00->_x;
					float y1000 = mass10->_y - mass00->_y;
					float z1000 = mass10->_z - mass00->_z;
					float x0100 = mass01->_x - mass00->_x;
					float y0100 = mass01->_y - mass00->_y;
					float z0100 = mass01->_z - mass00->_z;

					/*
					float x1011 = mass10->_x - mass11->_x;
					float y1011 = mass10->_y - mass11->_y;
					float z1011 = mass10->_z - mass11->_z;
					float x0111 = mass01->_x - mass11->_x;
					float y0111 = mass01->_y - mass11->_y;
					float z0111 = mass01->_z - mass11->_z;
					*/

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00->_x);
					_tri_positions.push_back(mass00->_y);
					_tri_positions.push_back(mass00->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass11->_x);
					_tri_positions.push_back(mass11->_y);
					_tri_positions.push_back(mass11->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
				}
				// back face
				{
					size_t i00 = cube->_start + i + j * x_size + (z_size-1) * xy_size;
					size_t i10 = cube->_start + (i+1) + j * x_size + (z_size-1) * xy_size;
					size_t i01 = cube->_start + i + (j+1) * x_size + (z_size-1) * xy_size;
					size_t i11 = cube->_start + (i+1) + (j+1) * x_size + (z_size-1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					float x1000 = mass10->_x - mass00->_x;
					float y1000 = mass10->_y - mass00->_y;
					float z1000 = mass10->_z - mass00->_z;
					float x0100 = mass01->_x - mass00->_x;
					float y0100 = mass01->_y - mass00->_y;
					float z0100 = mass01->_z - mass00->_z;

					/*
					float x1011 = mass10->_x - mass11->_x;
					float y1011 = mass10->_y - mass11->_y;
					float z1011 = mass10->_z - mass11->_z;
					float x0111 = mass01->_x - mass11->_x;
					float y0111 = mass01->_y - mass11->_y;
					float z0111 = mass01->_z - mass11->_z;
					*/

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00->_x);
					_tri_positions.push_back(mass00->_y);
					_tri_positions.push_back(mass00->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass11->_x);
					_tri_positions.push_back(mass11->_y);
					_tri_positions.push_back(mass11->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
				}
			}
		}

		// left and right face
		for (size_t i = 0; i < x_size-1; ++i) {
			for (size_t k = 0; k < z_size-1; ++k) {
				// left face
				{
					size_t i00 = cube->_start + i + k * xy_size;
					size_t i10 = cube->_start + (i+1) + k * xy_size;
					size_t i01 = cube->_start + i + (k+1) * xy_size;
					size_t i11 = cube->_start + (i+1) + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					float x1000 = mass10->_x - mass00->_x;
					float y1000 = mass10->_y - mass00->_y;
					float z1000 = mass10->_z - mass00->_z;
					float x0100 = mass01->_x - mass00->_x;
					float y0100 = mass01->_y - mass00->_y;
					float z0100 = mass01->_z - mass00->_z;

					/*
					float x1011 = mass10->_x - mass11->_x;
					float y1011 = mass10->_y - mass11->_y;
					float z1011 = mass10->_z - mass11->_z;
					float x0111 = mass01->_x - mass11->_x;
					float y0111 = mass01->_y - mass11->_y;
					float z0111 = mass01->_z - mass11->_z;
					*/

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00->_x);
					_tri_positions.push_back(mass00->_y);
					_tri_positions.push_back(mass00->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass11->_x);
					_tri_positions.push_back(mass11->_y);
					_tri_positions.push_back(mass11->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
				}
				// right face
				{
					size_t i00 = cube->_start + i + (y_size-1) * x_size + k * xy_size;
					size_t i10 = cube->_start + (i+1) + (y_size-1) * x_size + k * xy_size;
					size_t i01 = cube->_start + i + (y_size-1) * x_size + (k+1) * xy_size;
					size_t i11 = cube->_start + (i+1) + (y_size-1) * x_size + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					float x1000 = mass10->_x - mass00->_x;
					float y1000 = mass10->_y - mass00->_y;
					float z1000 = mass10->_z - mass00->_z;
					float x0100 = mass01->_x - mass00->_x;
					float y0100 = mass01->_y - mass00->_y;
					float z0100 = mass01->_z - mass00->_z;

					/*
					float x1011 = mass10->_x - mass11->_x;
					float y1011 = mass10->_y - mass11->_y;
					float z1011 = mass10->_z - mass11->_z;
					float x0111 = mass01->_x - mass11->_x;
					float y0111 = mass01->_y - mass11->_y;
					float z0111 = mass01->_z - mass11->_z;
					*/

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00->_x);
					_tri_positions.push_back(mass00->_y);
					_tri_positions.push_back(mass00->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass11->_x);
					_tri_positions.push_back(mass11->_y);
					_tri_positions.push_back(mass11->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
				}
			}
		}

		// top and bottom face
		for (size_t j = 0; j < y_size-1; ++j) {
			for (size_t k = 0; k < z_size-1; ++k) {
				// top face
				{
					size_t i00 = cube->_start + j*x_size + k * xy_size;
					size_t i10 = cube->_start + (j+1)*x_size + k * xy_size;
					size_t i01 = cube->_start + j*x_size + (k+1) * xy_size;
					size_t i11 = cube->_start + (j+1)*x_size + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					float x1000 = mass10->_x - mass00->_x;
					float y1000 = mass10->_y - mass00->_y;
					float z1000 = mass10->_z - mass00->_z;
					float x0100 = mass01->_x - mass00->_x;
					float y0100 = mass01->_y - mass00->_y;
					float z0100 = mass01->_z - mass00->_z;

					/*
					float x1011 = mass10->_x - mass11->_x;
					float y1011 = mass10->_y - mass11->_y;
					float z1011 = mass10->_z - mass11->_z;
					float x0111 = mass01->_x - mass11->_x;
					float y0111 = mass01->_y - mass11->_y;
					float z0111 = mass01->_z - mass11->_z;
					*/

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00->_x);
					_tri_positions.push_back(mass00->_y);
					_tri_positions.push_back(mass00->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass11->_x);
					_tri_positions.push_back(mass11->_y);
					_tri_positions.push_back(mass11->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
				}
				// top face
				{
					size_t i00 = cube->_start + x_size-1 + j*x_size + k * xy_size;
					size_t i10 = cube->_start + x_size-1 +(j+1)*x_size + k * xy_size;
					size_t i01 = cube->_start + x_size-1 +j*x_size + (k+1) * xy_size;
					size_t i11 = cube->_start + x_size-1 +(j+1)*x_size + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					float x1000 = mass10->_x - mass00->_x;
					float y1000 = mass10->_y - mass00->_y;
					float z1000 = mass10->_z - mass00->_z;
					float x0100 = mass01->_x - mass00->_x;
					float y0100 = mass01->_y - mass00->_y;
					float z0100 = mass01->_z - mass00->_z;

					/*
					float x1011 = mass10->_x - mass11->_x;
					float y1011 = mass10->_y - mass11->_y;
					float z1011 = mass10->_z - mass11->_z;
					float x0111 = mass01->_x - mass11->_x;
					float y0111 = mass01->_y - mass11->_y;
					float z0111 = mass01->_z - mass11->_z;
					*/

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00->_x);
					_tri_positions.push_back(mass00->_y);
					_tri_positions.push_back(mass00->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass01->_x);
					_tri_positions.push_back(mass01->_y);
					_tri_positions.push_back(mass01->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass10->_x);
					_tri_positions.push_back(mass10->_y);
					_tri_positions.push_back(mass10->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);

					_tri_positions.push_back(mass11->_x);
					_tri_positions.push_back(mass11->_y);
					_tri_positions.push_back(mass11->_z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
				}
			}
		}

		/*
		float threshold = 0.1;

		size_t ind_y = (_res_y+1);
		size_t ind_z = ind_y * (_res_z+1);
		size_t weights_size = (_res_x+1)*ind_z;
		std::vector<float> normals(weights_size*3);
		std::vector<float> weights(weights_size);
		for (unsigned int i = 0; i < weights_size; ++i) {
			normals[i*3  ] = 0.0;
			normals[i*3+1] = 0.0;
			normals[i*3+2] = 0.0;
			weights[i] = 0.0;
		}

		for (unsigned int i = cube->_start; i < cube->_end; i++) {
			Mass *mass = masses.getMass(i);
			int xi = static_cast<int>((mass->_x - min_x) * ux);
			int yi = static_cast<int>((mass->_y - min_y) * uy);
			int zi = static_cast<int>((mass->_z - min_z) * uz);
			
			int rx = mass->_radius * ux;
			int ry = mass->_radius * uy;
			int rz = mass->_radius * uz;

			for (int i = xi - rx; i <= xi + rx; ++i) {
				if (i < 0 || i > _res_x)
					continue;
				for (int j = yi - ry; j <= yi + ry; ++j) {
					if (j < 0 || j > _res_y)
						continue;
					for (int k = zi - rz; k <= zi + rz; ++k) {
						if (k < 0 || k > _res_z)
							continue;
						float x = min_x + static_cast<float>(i) * delta_x;
						float y = min_y + static_cast<float>(j) * delta_y;
						float z = min_z + static_cast<float>(k) * delta_z;
						size_t index = i + j*ind_y + k*ind_z;
						float dx = mass->_x - x;
						float dy = mass->_y - y;
						float dz = mass->_z - z;

						float r_sq = dx*dx + dy*dy + dz*dz;
						float inv_r = 1.0 / sqrt(r_sq);
						float weight = 1.0 / r_sq;
						float nx = dx * inv_r * weight;
						float ny = dy * inv_r * weight;
						float nz = dz * inv_r * weight;
						weights[index] += weight;
						normals[index*3  ] += nx;
						normals[index*3+1] += ny;
						normals[index*3+2] += nz;
					}
				}
			}

			for (size_t i = 0; i < _res_x; ++i) {
				for (size_t j = 0; j < _res_y; ++j) {
					for (size_t k = 0; k < _res_z; ++k) {
						size_t i000 = i + j*ind_y + k*ind_z;
						size_t i100 = (i+1) + j*ind_y + k*ind_z;
						size_t i010 = i + (j+1)*ind_y + k*ind_z;
						size_t i110 = (i+1) + (j+1)*ind_y + k*ind_z;
						size_t i001 = i + j*ind_y + (k+1)*ind_z;
						size_t i101 = (i+1) + j*ind_y + (k+1)*ind_z;
						size_t i011 = i + (j+1)*ind_y + (k+1)*ind_z;
						size_t i111 = (i+1) + (j+1)*ind_y + (k+1)*ind_z;

						// get cube corners
						float x000 = min_x + static_cast<float>(i)*delta_x;
						float y000 = min_y + static_cast<float>(j)*delta_y;
						float z000 = min_z + static_cast<float>(k)*delta_z;

						float x100 = min_x + static_cast<float>(i+1)*delta_x;
						float y100 = min_y + static_cast<float>(j)*delta_y;
						float z100 = min_z + static_cast<float>(k)*delta_z;

						float x010 = min_x + static_cast<float>(i)*delta_x;
						float y010 = min_y + static_cast<float>(j+1)*delta_y;
						float z010 = min_z + static_cast<float>(k)*delta_z;

						float x110 = min_x + static_cast<float>(i+1)*delta_x;
						float y110 = min_y + static_cast<float>(j+1)*delta_y;
						float z110 = min_z + static_cast<float>(k)*delta_z;

						float x001 = min_x + static_cast<float>(i)*delta_x;
						float y001 = min_y + static_cast<float>(j)*delta_y;
						float z001 = min_z + static_cast<float>(k+1)*delta_z;

						float x101 = min_x + static_cast<float>(i+1)*delta_x;
						float y101 = min_y + static_cast<float>(j)*delta_y;
						float z101 = min_z + static_cast<float>(k+1)*delta_z;

						float x011 = min_x + static_cast<float>(i)*delta_x;
						float y011 = min_y + static_cast<float>(j+1)*delta_y;
						float z011 = min_z + static_cast<float>(k+1)*delta_z;

						float x111 = min_x + static_cast<float>(i+1)*delta_x;
						float y111 = min_y + static_cast<float>(j+1)*delta_y;
						float z111 = min_z + static_cast<float>(k+1)*delta_z;
						
						float n000_x = normals[i000*3];
						float n000_y = normals[i000*3+1];
						float n000_z = normals[i000*3+2];
						float n100_x = normals[i100*3];
						float n100_y = normals[i100*3+1];
						float n100_z = normals[i100*3+2];
						float n010_x = normals[i010*3];
						float n010_y = normals[i010*3+1];
						float n010_z = normals[i010*3+2];
						float n110_x = normals[i110*3];
						float n110_y = normals[i110*3+1];
						float n110_z = normals[i110*3+2];
						float n001_x = normals[i001*3];
						float n001_y = normals[i001*3+1];
						float n001_z = normals[i001*3+2];
						float n101_x = normals[i101*3];
						float n101_y = normals[i101*3+1];
						float n101_z = normals[i101*3+2];
						float n011_x = normals[i011*3];
						float n011_y = normals[i011*3+1];
						float n011_z = normals[i011*3+2];
						float n111_x = normals[i111*3];
						float n111_y = normals[i111*3+1];
						float n111_z = normals[i111*3+2];

						// get weights and normals

						// tetrahedron 0
						compute_triangles(
							threshold,
							i000, x000, y000, z000, n000_x, n000_y, n000_z,
							i100, x100, y100, z100, n100_x, n100_y, n100_z,
							i010, x010, y010, z010, n010_x, n010_y, n010_z,
							i011, x011, y011, z011, n011_x, n011_y, n011_z,
							weights,
							_tri_positions,
							_tri_normals);
						// tetrahedron 1
						compute_triangles(
							threshold,
							i100, x100, y100, z100, n100_x, n100_y, n100_z,
							i010, x010, y010, z010, n010_x, n010_y, n010_z,
							i011, x011, y011, z011, n011_x, n011_y, n011_z,
							i110, x110, y110, z110, n110_x, n110_y, n110_z,
							weights,
							_tri_positions,
							_tri_normals);
						// tetrahedron 2
						compute_triangles(
							threshold,
							i000, x000, y000, z000, n000_x, n000_y, n000_z,
							i100, x100, y100, z100, n100_x, n100_y, n100_z,
							i001, x001, y001, z001, n001_x, n001_y, n001_z,
							i011, x011, y011, z011, n011_x, n011_y, n011_z,
							weights,
							_tri_positions,
							_tri_normals);
						// tetrahedron 3
						compute_triangles(
							threshold,
							i100, x100, y100, z100, n100_x, n100_y, n100_z,
							i110, x110, y110, z110, n110_x, n110_y, n110_z,
							i011, x011, y011, z011, n011_x, n011_y, n011_z,
							i111, x111, y111, z111, n111_x, n111_y, n111_z,
							weights,
							_tri_positions,
							_tri_normals);
						// tetrahedron 4
						compute_triangles(
							threshold,
							i001, x001, y001, z001, n001_x, n001_y, n001_z,
							i100, x100, y100, z100, n100_x, n100_y, n100_z,
							i101, x101, y101, z101, n101_x, n101_y, n101_z,
							i011, x011, y011, z011, n011_x, n011_y, n011_z,
							weights,
							_tri_positions,
							_tri_normals);
						// tetrahedron 5
						compute_triangles(
							threshold,
							i100, x100, y100, z100, n100_x, n100_y, n100_z,
							i101, x101, y101, z101, n101_x, n101_y, n101_z,
							i011, x011, y011, z011, n011_x, n011_y, n011_z,
							i111, x111, y111, z111, n111_x, n111_y, n111_z,
							weights,
							_tri_positions,
							_tri_normals);
					}
				}
			}
		}
		*/
	}

	// buffers
	if (_tri_positions.size() > 0 && _tri_normals.size() > 0) {
		//std::cout << "Generate cube buffers" << std::endl;
		if (_cube_pos_buffer == 0) {
			// generate GL buffer
			glGenBuffers(1, &_cube_pos_buffer);
		}
		glBindBuffer(GL_ARRAY_BUFFER, _cube_pos_buffer);
		//std::cout << "Fill pos buffer" << std::endl;
		glBufferData(
			GL_ARRAY_BUFFER,
			_tri_positions.size()*sizeof(float),
			&_tri_positions[0],
			GL_DYNAMIC_DRAW);
		if (_cube_norm_buffer == 0) {
			// generate GL buffer
			glGenBuffers(1, &_cube_norm_buffer);
		}
		glBindBuffer(GL_ARRAY_BUFFER, _cube_norm_buffer);
		//std::cout << "Fill normals buffer" << std::endl;
		glBufferData(
			GL_ARRAY_BUFFER,
			_tri_normals.size()*sizeof(float),
			&_tri_normals[0],
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// vertex array
		//std::cout << "Generate cube array" << std::endl;
		if (_cube_array == 0) {
			glGenVertexArrays(1, &_cube_array);
		}
		glBindVertexArray(_cube_array);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, _cube_pos_buffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, _cube_norm_buffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindVertexArray(0);
		//std::cout << "Done generating cube array" << std::endl;
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

void SigAsiaDemo::CubeList::render(
	glm::mat4 ModelView,
	glm::mat4 Projection,
	glm::mat3 Normal) const
{
	if (_cube_program == 0) {
		std::cerr << "Warning: _cube_program not set." \
		<< std::endl;
		return;
	}
	if (_cube_ModelViewLocation == -1) {
		std::cerr << "Warning: _cube_ModelViewLocation not set." \
		<< std::endl;
		return;
	}
	if (_cube_ProjectionLocation == -1) {
		std::cerr << "Warning: _cube_ProjectionLocation not set." \
		<< std::endl;
		return;
	}
	if (_cube_NormalLocation == -1) {
		std::cerr << "Warning: _cube_NormalLocation not set." \
		<< std::endl;
		return;
	}
	if (_tri_positions.empty()) {
		std::cerr << "Warning: positions not set." \
		<< std::endl;
		return;
	}
	if (_tri_normals.empty()) {
		std::cerr << "Warning: normals not set." \
		<< std::endl;
		return;
	}
	if (_cube_array == 0) {
		std::cerr << "Warning: _cube_array not set." \
		<< std::endl;
		return;
	}

	// bind depth shader
	glUseProgram(_cube_program);

	// setup uniforms
	glUniformMatrix4fv(
		_cube_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_cube_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));
	glUniformMatrix3fv(
		_cube_NormalLocation,
		1, GL_FALSE,
		glm::value_ptr(Normal));

	glBindVertexArray(_cube_array);
	glDrawArrays(GL_TRIANGLES, 0, _tri_positions.size()/3);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(0);
}
