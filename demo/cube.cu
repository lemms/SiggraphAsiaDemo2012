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
		_spring_start(0),
		_spring_end(0),
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
	_spring_start = springs.size();
	std::cout << "Starting at mass index " << _start << "." << std::endl;
	std::cout << "Starting at spring index " << _spring_start << "." << std::endl;

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
	std::cout << "Ending at mass index " << _end << "." << std::endl;

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

	_spring_end = springs.size();
	std::cout << "Ending at spring index " << _spring_end << "." << std::endl;
}

SigAsiaDemo::CubeList::CubeList(
	float ks,
	float kd,
	unsigned int threads) :
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
		_cube_norm_buffer(0),
		_cube_col_buffer(0)
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
	masses.download();

	_tri_positions.clear();
	_tri_normals.clear();
	_tri_colors.clear();
	for (std::vector<Cube>::iterator cube = _cubes.begin();
		cube != _cubes.end(); cube++) {
		float centroid_x = 0.0;
		float centroid_y = 0.0;
		float centroid_z = 0.0;
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

			centroid_x += mass->_x;
			centroid_y += mass->_y;
			centroid_z += mass->_z;
		}

		float inv_size = 1.0 / static_cast<float>(cube->_end - cube->_start);
		centroid_x *= inv_size;
		centroid_y *= inv_size;
		centroid_z *= inv_size;

		cube->_min_x = min_x;
		cube->_min_y = min_y;
		cube->_min_z = min_z;

		cube->_max_x = max_x;
		cube->_max_y = max_y;
		cube->_max_z = max_z;

		// generate surface mesh

		float plane_size = 500.0;
		_tri_positions.push_back(-plane_size);
		_tri_positions.push_back(0.0);
		_tri_positions.push_back(-plane_size);
		_tri_normals.push_back(0.0); _tri_normals.push_back(1.0); _tri_normals.push_back(0.0);
		_tri_colors.push_back(0.0); _tri_colors.push_back(1.0); _tri_colors.push_back(0.0);

		_tri_positions.push_back(plane_size);
		_tri_positions.push_back(0.0);
		_tri_positions.push_back(-plane_size);
		_tri_normals.push_back(0.0); _tri_normals.push_back(1.0); _tri_normals.push_back(0.0);
		_tri_colors.push_back(0.0); _tri_colors.push_back(1.0); _tri_colors.push_back(0.0);

		_tri_positions.push_back(-plane_size);
		_tri_positions.push_back(0.0);
		_tri_positions.push_back(plane_size);
		_tri_normals.push_back(0.0); _tri_normals.push_back(1.0); _tri_normals.push_back(0.0);
		_tri_colors.push_back(0.0); _tri_colors.push_back(1.0); _tri_colors.push_back(0.0);

		_tri_positions.push_back(-plane_size);
		_tri_positions.push_back(0.0);
		_tri_positions.push_back(plane_size);
		_tri_normals.push_back(0.0); _tri_normals.push_back(1.0); _tri_normals.push_back(0.0);
		_tri_colors.push_back(0.0); _tri_colors.push_back(1.0); _tri_colors.push_back(0.0);

		_tri_positions.push_back(plane_size);
		_tri_positions.push_back(0.0);
		_tri_positions.push_back(-plane_size);
		_tri_normals.push_back(0.0); _tri_normals.push_back(1.0); _tri_normals.push_back(0.0);
		_tri_colors.push_back(0.0); _tri_colors.push_back(1.0); _tri_colors.push_back(0.0);

		_tri_positions.push_back(plane_size);
		_tri_positions.push_back(0.0);
		_tri_positions.push_back(plane_size);
		_tri_normals.push_back(0.0); _tri_normals.push_back(1.0); _tri_normals.push_back(0.0);
		_tri_colors.push_back(0.0); _tri_colors.push_back(1.0); _tri_colors.push_back(0.0);

		float cube_scale = 1.3;

		size_t x_size = cube->_size_x+1;
		size_t y_size = cube->_size_y+1;
		size_t z_size = cube->_size_z+1;
		size_t xy_size = x_size * y_size;

		// front and back face
		for (size_t i = 0; i < x_size-1; ++i) {
			for (size_t j = 0; j < y_size-1; ++j) {
				// front face
				size_t i00 = cube->_start + i + j * x_size;
				size_t i10 = cube->_start + (i+1) + j * x_size;
				size_t i01 = cube->_start + i + (j+1) * x_size;
				size_t i11 = cube->_start + (i+1) + (j+1) * x_size;
				Mass *mass00 = masses.getMass(i00);
				Mass *mass10 = masses.getMass(i10);
				Mass *mass01 = masses.getMass(i01);
				Mass *mass11 = masses.getMass(i11);

				// vector from centroid to mass00
				float v00x = (mass00->_x - centroid_x);
				float v00y = (mass00->_y - centroid_y);
				float v00z = (mass00->_z - centroid_z);
				float v00_l = 1.0 / sqrt(v00x*v00x + v00y*v00y + v00z*v00z);
				// normalize and scale by radius
				v00x *= v00_l * cube->_radius * cube_scale;
				v00y *= v00_l * cube->_radius * cube_scale;
				v00z *= v00_l * cube->_radius * cube_scale;
				float mass00x = mass00->_x + v00x;
				float mass00y = mass00->_y + v00y;
				float mass00z = mass00->_z + v00z;

				// vector from centroid to mass01
				float v01x = (mass01->_x - centroid_x);
				float v01y = (mass01->_y - centroid_y);
				float v01z = (mass01->_z - centroid_z);
				float v01_l = 1.0 / sqrt(v01x*v01x + v01y*v01y + v01z*v01z);
				// normalize and scale by radius
				v01x *= v01_l * cube->_radius * cube_scale;
				v01y *= v01_l * cube->_radius * cube_scale;
				v01z *= v01_l * cube->_radius * cube_scale;
				float mass01x = mass01->_x + v01x;
				float mass01y = mass01->_y + v01y;
				float mass01z = mass01->_z + v01z;

				// vector from centroid to mass10
				float v10x = (mass10->_x - centroid_x);
				float v10y = (mass10->_y - centroid_y);
				float v10z = (mass10->_z - centroid_z);
				float v10_l = 1.0 / sqrt(v10x*v10x + v10y*v10y + v10z*v10z);
				// normalize and scale by radius
				v10x *= v10_l * cube->_radius * cube_scale;
				v10y *= v10_l * cube->_radius * cube_scale;
				v10z *= v10_l * cube->_radius * cube_scale;
				float mass10x = mass10->_x + v10x;
				float mass10y = mass10->_y + v10y;
				float mass10z = mass10->_z + v10z;

				// vector from centroid to mass11
				float v11x = (mass11->_x - centroid_x);
				float v11y = (mass11->_y - centroid_y);
				float v11z = (mass11->_z - centroid_z);
				float v11_l = 1.0 / sqrt(v11x*v11x + v11y*v11y + v11z*v11z);
				// normalize and scale by radius
				v11x *= v11_l * cube->_radius * cube_scale;
				v11y *= v11_l * cube->_radius * cube_scale;
				v11z *= v11_l * cube->_radius * cube_scale;
				float mass11x = mass11->_x + v11x;
				float mass11y = mass11->_y + v11y;
				float mass11z = mass11->_z + v11z;

				float x1000 = mass10x - mass00x;
				float y1000 = mass10y - mass00y;
				float z1000 = mass10z - mass00z;
				float x0100 = mass01x - mass00x;
				float y0100 = mass01y - mass00y;
				float z0100 = mass01z - mass00z;

				float nx = -(y1000*z0100 - y0100*z1000);
				float ny = x1000*z0100 - x0100*z1000;
				float nz = -(x1000*y0100 - x0100*y1000);
				float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

				_tri_positions.push_back(mass00x);
				_tri_positions.push_back(mass00y);
				_tri_positions.push_back(mass00z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass10x);
				_tri_positions.push_back(mass10y);
				_tri_positions.push_back(mass10z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass01x);
				_tri_positions.push_back(mass01y);
				_tri_positions.push_back(mass01z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass01x);
				_tri_positions.push_back(mass01y);
				_tri_positions.push_back(mass01z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass10x);
				_tri_positions.push_back(mass10y);
				_tri_positions.push_back(mass10z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass11x);
				_tri_positions.push_back(mass11y);
				_tri_positions.push_back(mass11z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);
			}
		}

		for (size_t i = 0; i < x_size-1; ++i) {
			for (size_t j = 0; j < y_size-1; ++j) {
				// back face
				size_t i00 = cube->_start + i + j * x_size + (z_size-1) * xy_size;
				size_t i10 = cube->_start + (i+1) + j * x_size + (z_size-1) * xy_size;
				size_t i01 = cube->_start + i + (j+1) * x_size + (z_size-1) * xy_size;
				size_t i11 = cube->_start + (i+1) + (j+1) * x_size + (z_size-1) * xy_size;
				Mass *mass00 = masses.getMass(i00);
				Mass *mass10 = masses.getMass(i10);
				Mass *mass01 = masses.getMass(i01);
				Mass *mass11 = masses.getMass(i11);

				// vector from centroid to mass00
				float v00x = (mass00->_x - centroid_x);
				float v00y = (mass00->_y - centroid_y);
				float v00z = (mass00->_z - centroid_z);
				float v00_l = 1.0 / sqrt(v00x*v00x + v00y*v00y + v00z*v00z);
				// normalize and scale by radius
				v00x *= v00_l * cube->_radius * cube_scale;
				v00y *= v00_l * cube->_radius * cube_scale;
				v00z *= v00_l * cube->_radius * cube_scale;
				float mass00x = mass00->_x + v00x;
				float mass00y = mass00->_y + v00y;
				float mass00z = mass00->_z + v00z;

				// vector from centroid to mass01
				float v01x = (mass01->_x - centroid_x);
				float v01y = (mass01->_y - centroid_y);
				float v01z = (mass01->_z - centroid_z);
				float v01_l = 1.0 / sqrt(v01x*v01x + v01y*v01y + v01z*v01z);
				// normalize and scale by radius
				v01x *= v01_l * cube->_radius * cube_scale;
				v01y *= v01_l * cube->_radius * cube_scale;
				v01z *= v01_l * cube->_radius * cube_scale;
				float mass01x = mass01->_x + v01x;
				float mass01y = mass01->_y + v01y;
				float mass01z = mass01->_z + v01z;

				// vector from centroid to mass10
				float v10x = (mass10->_x - centroid_x);
				float v10y = (mass10->_y - centroid_y);
				float v10z = (mass10->_z - centroid_z);
				float v10_l = 1.0 / sqrt(v10x*v10x + v10y*v10y + v10z*v10z);
				// normalize and scale by radius
				v10x *= v10_l * cube->_radius * cube_scale;
				v10y *= v10_l * cube->_radius * cube_scale;
				v10z *= v10_l * cube->_radius * cube_scale;
				float mass10x = mass10->_x + v10x;
				float mass10y = mass10->_y + v10y;
				float mass10z = mass10->_z + v10z;

				// vector from centroid to mass11
				float v11x = (mass11->_x - centroid_x);
				float v11y = (mass11->_y - centroid_y);
				float v11z = (mass11->_z - centroid_z);
				float v11_l = 1.0 / sqrt(v11x*v11x + v11y*v11y + v11z*v11z);
				// normalize and scale by radius
				v11x *= v11_l * cube->_radius * cube_scale;
				v11y *= v11_l * cube->_radius * cube_scale;
				v11z *= v11_l * cube->_radius * cube_scale;
				float mass11x = mass11->_x + v11x;
				float mass11y = mass11->_y + v11y;
				float mass11z = mass11->_z + v11z;

				float x1000 = mass10x - mass00x;
				float y1000 = mass10y - mass00y;
				float z1000 = mass10z - mass00z;
				float x0100 = mass01x - mass00x;
				float y0100 = mass01y - mass00y;
				float z0100 = mass01z - mass00z;

				float nx = y1000*z0100 - y0100*z1000;
				float ny = -(x1000*z0100 - x0100*z1000);
				float nz = x1000*y0100 - x0100*y1000;
				float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

				_tri_positions.push_back(mass00x);
				_tri_positions.push_back(mass00y);
				_tri_positions.push_back(mass00z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass10x);
				_tri_positions.push_back(mass10y);
				_tri_positions.push_back(mass10z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass01x);
				_tri_positions.push_back(mass01y);
				_tri_positions.push_back(mass01z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass01x);
				_tri_positions.push_back(mass01y);
				_tri_positions.push_back(mass01z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass10x);
				_tri_positions.push_back(mass10y);
				_tri_positions.push_back(mass10z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);

				_tri_positions.push_back(mass11x);
				_tri_positions.push_back(mass11y);
				_tri_positions.push_back(mass11z);
				_tri_normals.push_back(nx*l_n);
				_tri_normals.push_back(ny*l_n);
				_tri_normals.push_back(nz*l_n);
				_tri_colors.push_back(1.0);
				_tri_colors.push_back(0.0);
				_tri_colors.push_back(0.0);
			}
		}

		// top and bottom face
		for (size_t i = 0; i < x_size-1; ++i) {
			for (size_t k = 0; k < z_size-1; ++k) {
				// top face
				{
					size_t i00 = cube->_start + i + k * xy_size;
					size_t i10 = cube->_start + (i+1) + k * xy_size;
					size_t i01 = cube->_start + i + (k+1) * xy_size;
					size_t i11 = cube->_start + (i+1) + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					// vector from centroid to mass00
					float v00x = (mass00->_x - centroid_x);
					float v00y = (mass00->_y - centroid_y);
					float v00z = (mass00->_z - centroid_z);
					float v00_l = 1.0 / sqrt(v00x*v00x + v00y*v00y + v00z*v00z);
					// normalize and scale by radius
					v00x *= v00_l * cube->_radius * cube_scale;
					v00y *= v00_l * cube->_radius * cube_scale;
					v00z *= v00_l * cube->_radius * cube_scale;
					float mass00x = mass00->_x + v00x;
					float mass00y = mass00->_y + v00y;
					float mass00z = mass00->_z + v00z;

					// vector from centroid to mass01
					float v01x = (mass01->_x - centroid_x);
					float v01y = (mass01->_y - centroid_y);
					float v01z = (mass01->_z - centroid_z);
					float v01_l = 1.0 / sqrt(v01x*v01x + v01y*v01y + v01z*v01z);
					// normalize and scale by radius
					v01x *= v01_l * cube->_radius * cube_scale;
					v01y *= v01_l * cube->_radius * cube_scale;
					v01z *= v01_l * cube->_radius * cube_scale;
					float mass01x = mass01->_x + v01x;
					float mass01y = mass01->_y + v01y;
					float mass01z = mass01->_z + v01z;

					// vector from centroid to mass10
					float v10x = (mass10->_x - centroid_x);
					float v10y = (mass10->_y - centroid_y);
					float v10z = (mass10->_z - centroid_z);
					float v10_l = 1.0 / sqrt(v10x*v10x + v10y*v10y + v10z*v10z);
					// normalize and scale by radius
					v10x *= v10_l * cube->_radius * cube_scale;
					v10y *= v10_l * cube->_radius * cube_scale;
					v10z *= v10_l * cube->_radius * cube_scale;
					float mass10x = mass10->_x + v10x;
					float mass10y = mass10->_y + v10y;
					float mass10z = mass10->_z + v10z;

					// vector from centroid to mass11
					float v11x = (mass11->_x - centroid_x);
					float v11y = (mass11->_y - centroid_y);
					float v11z = (mass11->_z - centroid_z);
					float v11_l = 1.0 / sqrt(v11x*v11x + v11y*v11y + v11z*v11z);
					// normalize and scale by radius
					v11x *= v11_l * cube->_radius * cube_scale;
					v11y *= v11_l * cube->_radius * cube_scale;
					v11z *= v11_l * cube->_radius * cube_scale;
					float mass11x = mass11->_x + v11x;
					float mass11y = mass11->_y + v11y;
					float mass11z = mass11->_z + v11z;

					float x1000 = mass10x - mass00x;
					float y1000 = mass10y - mass00y;
					float z1000 = mass10z - mass00z;
					float x0100 = mass01x - mass00x;
					float y0100 = mass01y - mass00y;
					float z0100 = mass01z - mass00z;

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00x);
					_tri_positions.push_back(mass00y);
					_tri_positions.push_back(mass00z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass11x);
					_tri_positions.push_back(mass11y);
					_tri_positions.push_back(mass11z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);
				}
				// bottom face
				{
					size_t i00 = cube->_start + i + (y_size-1) * x_size + k * xy_size;
					size_t i10 = cube->_start + (i+1) + (y_size-1) * x_size + k * xy_size;
					size_t i01 = cube->_start + i + (y_size-1) * x_size + (k+1) * xy_size;
					size_t i11 = cube->_start + (i+1) + (y_size-1) * x_size + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					// vector from centroid to mass00
					float v00x = (mass00->_x - centroid_x);
					float v00y = (mass00->_y - centroid_y);
					float v00z = (mass00->_z - centroid_z);
					float v00_l = 1.0 / sqrt(v00x*v00x + v00y*v00y + v00z*v00z);
					// normalize and scale by radius
					v00x *= v00_l * cube->_radius * cube_scale;
					v00y *= v00_l * cube->_radius * cube_scale;
					v00z *= v00_l * cube->_radius * cube_scale;
					float mass00x = mass00->_x + v00x;
					float mass00y = mass00->_y + v00y;
					float mass00z = mass00->_z + v00z;

					// vector from centroid to mass01
					float v01x = (mass01->_x - centroid_x);
					float v01y = (mass01->_y - centroid_y);
					float v01z = (mass01->_z - centroid_z);
					float v01_l = 1.0 / sqrt(v01x*v01x + v01y*v01y + v01z*v01z);
					// normalize and scale by radius
					v01x *= v01_l * cube->_radius * cube_scale;
					v01y *= v01_l * cube->_radius * cube_scale;
					v01z *= v01_l * cube->_radius * cube_scale;
					float mass01x = mass01->_x + v01x;
					float mass01y = mass01->_y + v01y;
					float mass01z = mass01->_z + v01z;

					// vector from centroid to mass10
					float v10x = (mass10->_x - centroid_x);
					float v10y = (mass10->_y - centroid_y);
					float v10z = (mass10->_z - centroid_z);
					float v10_l = 1.0 / sqrt(v10x*v10x + v10y*v10y + v10z*v10z);
					// normalize and scale by radius
					v10x *= v10_l * cube->_radius * cube_scale;
					v10y *= v10_l * cube->_radius * cube_scale;
					v10z *= v10_l * cube->_radius * cube_scale;
					float mass10x = mass10->_x + v10x;
					float mass10y = mass10->_y + v10y;
					float mass10z = mass10->_z + v10z;

					// vector from centroid to mass11
					float v11x = (mass11->_x - centroid_x);
					float v11y = (mass11->_y - centroid_y);
					float v11z = (mass11->_z - centroid_z);
					float v11_l = 1.0 / sqrt(v11x*v11x + v11y*v11y + v11z*v11z);
					// normalize and scale by radius
					v11x *= v11_l * cube->_radius * cube_scale;
					v11y *= v11_l * cube->_radius * cube_scale;
					v11z *= v11_l * cube->_radius * cube_scale;
					float mass11x = mass11->_x + v11x;
					float mass11y = mass11->_y + v11y;
					float mass11z = mass11->_z + v11z;

					float x1000 = mass10x - mass00x;
					float y1000 = mass10y - mass00y;
					float z1000 = mass10z - mass00z;
					float x0100 = mass01x - mass00x;
					float y0100 = mass01y - mass00y;
					float z0100 = mass01z - mass00z;

					float nx = -(y1000*z0100 - y0100*z1000);
					float ny = x1000*z0100 - x0100*z1000;
					float nz = -(x1000*y0100 - x0100*y1000);
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00x);
					_tri_positions.push_back(mass00y);
					_tri_positions.push_back(mass00z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass11x);
					_tri_positions.push_back(mass11y);
					_tri_positions.push_back(mass11z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);
				}
			}
		}

		// left and right face
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

					// vector from centroid to mass00
					float v00x = (mass00->_x - centroid_x);
					float v00y = (mass00->_y - centroid_y);
					float v00z = (mass00->_z - centroid_z);
					float v00_l = 1.0 / sqrt(v00x*v00x + v00y*v00y + v00z*v00z);
					// normalize and scale by radius
					v00x *= v00_l * cube->_radius * cube_scale;
					v00y *= v00_l * cube->_radius * cube_scale;
					v00z *= v00_l * cube->_radius * cube_scale;
					float mass00x = mass00->_x + v00x;
					float mass00y = mass00->_y + v00y;
					float mass00z = mass00->_z + v00z;

					// vector from centroid to mass01
					float v01x = (mass01->_x - centroid_x);
					float v01y = (mass01->_y - centroid_y);
					float v01z = (mass01->_z - centroid_z);
					float v01_l = 1.0 / sqrt(v01x*v01x + v01y*v01y + v01z*v01z);
					// normalize and scale by radius
					v01x *= v01_l * cube->_radius * cube_scale;
					v01y *= v01_l * cube->_radius * cube_scale;
					v01z *= v01_l * cube->_radius * cube_scale;
					float mass01x = mass01->_x + v01x;
					float mass01y = mass01->_y + v01y;
					float mass01z = mass01->_z + v01z;

					// vector from centroid to mass10
					float v10x = (mass10->_x - centroid_x);
					float v10y = (mass10->_y - centroid_y);
					float v10z = (mass10->_z - centroid_z);
					float v10_l = 1.0 / sqrt(v10x*v10x + v10y*v10y + v10z*v10z);
					// normalize and scale by radius
					v10x *= v10_l * cube->_radius * cube_scale;
					v10y *= v10_l * cube->_radius * cube_scale;
					v10z *= v10_l * cube->_radius * cube_scale;
					float mass10x = mass10->_x + v10x;
					float mass10y = mass10->_y + v10y;
					float mass10z = mass10->_z + v10z;

					// vector from centroid to mass11
					float v11x = (mass11->_x - centroid_x);
					float v11y = (mass11->_y - centroid_y);
					float v11z = (mass11->_z - centroid_z);
					float v11_l = 1.0 / sqrt(v11x*v11x + v11y*v11y + v11z*v11z);
					// normalize and scale by radius
					v11x *= v11_l * cube->_radius * cube_scale;
					v11y *= v11_l * cube->_radius * cube_scale;
					v11z *= v11_l * cube->_radius * cube_scale;
					float mass11x = mass11->_x + v11x;
					float mass11y = mass11->_y + v11y;
					float mass11z = mass11->_z + v11z;

					float x1000 = mass10x - mass00x;
					float y1000 = mass10y - mass00y;
					float z1000 = mass10z - mass00z;
					float x0100 = mass01x - mass00x;
					float y0100 = mass01y - mass00y;
					float z0100 = mass01z - mass00z;

					float nx = y1000*z0100 - y0100*z1000;
					float ny = -(x1000*z0100 - x0100*z1000);
					float nz = x1000*y0100 - x0100*y1000;
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00x);
					_tri_positions.push_back(mass00y);
					_tri_positions.push_back(mass00z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass11x);
					_tri_positions.push_back(mass11y);
					_tri_positions.push_back(mass11z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);
				}
				// right face
				{
					size_t i00 = cube->_start + x_size-1 + j*x_size + k * xy_size;
					size_t i10 = cube->_start + x_size-1 +(j+1)*x_size + k * xy_size;
					size_t i01 = cube->_start + x_size-1 +j*x_size + (k+1) * xy_size;
					size_t i11 = cube->_start + x_size-1 +(j+1)*x_size + (k+1) * xy_size;
					Mass *mass00 = masses.getMass(i00);
					Mass *mass10 = masses.getMass(i10);
					Mass *mass01 = masses.getMass(i01);
					Mass *mass11 = masses.getMass(i11);

					// vector from centroid to mass00
					float v00x = (mass00->_x - centroid_x);
					float v00y = (mass00->_y - centroid_y);
					float v00z = (mass00->_z - centroid_z);
					float v00_l = 1.0 / sqrt(v00x*v00x + v00y*v00y + v00z*v00z);
					// normalize and scale by radius
					v00x *= v00_l * cube->_radius * cube_scale;
					v00y *= v00_l * cube->_radius * cube_scale;
					v00z *= v00_l * cube->_radius * cube_scale;
					float mass00x = mass00->_x + v00x;
					float mass00y = mass00->_y + v00y;
					float mass00z = mass00->_z + v00z;

					// vector from centroid to mass01
					float v01x = (mass01->_x - centroid_x);
					float v01y = (mass01->_y - centroid_y);
					float v01z = (mass01->_z - centroid_z);
					float v01_l = 1.0 / sqrt(v01x*v01x + v01y*v01y + v01z*v01z);
					// normalize and scale by radius
					v01x *= v01_l * cube->_radius * cube_scale;
					v01y *= v01_l * cube->_radius * cube_scale;
					v01z *= v01_l * cube->_radius * cube_scale;
					float mass01x = mass01->_x + v01x;
					float mass01y = mass01->_y + v01y;
					float mass01z = mass01->_z + v01z;

					// vector from centroid to mass10
					float v10x = (mass10->_x - centroid_x);
					float v10y = (mass10->_y - centroid_y);
					float v10z = (mass10->_z - centroid_z);
					float v10_l = 1.0 / sqrt(v10x*v10x + v10y*v10y + v10z*v10z);
					// normalize and scale by radius
					v10x *= v10_l * cube->_radius * cube_scale;
					v10y *= v10_l * cube->_radius * cube_scale;
					v10z *= v10_l * cube->_radius * cube_scale;
					float mass10x = mass10->_x + v10x;
					float mass10y = mass10->_y + v10y;
					float mass10z = mass10->_z + v10z;

					// vector from centroid to mass11
					float v11x = (mass11->_x - centroid_x);
					float v11y = (mass11->_y - centroid_y);
					float v11z = (mass11->_z - centroid_z);
					float v11_l = 1.0 / sqrt(v11x*v11x + v11y*v11y + v11z*v11z);
					// normalize and scale by radius
					v11x *= v11_l * cube->_radius * cube_scale;
					v11y *= v11_l * cube->_radius * cube_scale;
					v11z *= v11_l * cube->_radius * cube_scale;
					float mass11x = mass11->_x + v11x;
					float mass11y = mass11->_y + v11y;
					float mass11z = mass11->_z + v11z;

					float x1000 = mass10x - mass00x;
					float y1000 = mass10y - mass00y;
					float z1000 = mass10z - mass00z;
					float x0100 = mass01x - mass00x;
					float y0100 = mass01y - mass00y;
					float z0100 = mass01z - mass00z;

					float nx = -(y1000*z0100 - y0100*z1000);
					float ny = x1000*z0100 - x0100*z1000;
					float nz = -(x1000*y0100 - x0100*y1000);
					float l_n = 1.0/sqrt(nx*nx + ny*ny + nz*nz);

					_tri_positions.push_back(mass00x);
					_tri_positions.push_back(mass00y);
					_tri_positions.push_back(mass00z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass01x);
					_tri_positions.push_back(mass01y);
					_tri_positions.push_back(mass01z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass10x);
					_tri_positions.push_back(mass10y);
					_tri_positions.push_back(mass10z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);

					_tri_positions.push_back(mass11x);
					_tri_positions.push_back(mass11y);
					_tri_positions.push_back(mass11z);
					_tri_normals.push_back(nx*l_n);
					_tri_normals.push_back(ny*l_n);
					_tri_normals.push_back(nz*l_n);
					_tri_colors.push_back(1.0);
					_tri_colors.push_back(0.0);
					_tri_colors.push_back(0.0);
				}
			}
		}
	}

	// buffers
	if (_tri_positions.size() > 0 && _tri_normals.size() > 0) {
		if (_cube_pos_buffer == 0) {
			// generate GL buffer
			glGenBuffers(1, &_cube_pos_buffer);
		}
		glBindBuffer(GL_ARRAY_BUFFER, _cube_pos_buffer);
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
		glBufferData(
			GL_ARRAY_BUFFER,
			_tri_normals.size()*sizeof(float),
			&_tri_normals[0],
			GL_DYNAMIC_DRAW);

		if (_cube_col_buffer == 0) {
			// generate GL buffer
			glGenBuffers(1, &_cube_col_buffer);
		}
		glBindBuffer(GL_ARRAY_BUFFER, _cube_col_buffer);
		glBufferData(
			GL_ARRAY_BUFFER,
			_tri_colors.size()*sizeof(float),
			&_tri_colors[0],
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// vertex array
		if (_cube_array == 0) {
			glGenVertexArrays(1, &_cube_array);
		}
		glBindVertexArray(_cube_array);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glBindBuffer(GL_ARRAY_BUFFER, _cube_pos_buffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, _cube_norm_buffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, _cube_col_buffer);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindVertexArray(0);
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
			} else {
				// overlap

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
