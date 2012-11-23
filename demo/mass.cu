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

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

// GLM
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "mass.h"

SigAsiaDemo::Mass::Mass(
	float mass,
	float x,
	float y,
	float z,
	float fx,
	float fy,
	float fz,
	int state,
	float radius) :
		_mass(mass),
		_x(x), _y(y), _z(z),
		_vx(0.0), _vy(0.0), _vz(0.0),
		_tx(x), _ty(y), _tz(z),
		_tvx(0.0), _tvy(0.0), _tvz(0.0),
		_fx(fx), _fy(fy), _fz(fz),
		_k1x(0.0), _k1y(0.0), _k1z(0.0),
		_k2x(0.0), _k2y(0.0), _k2z(0.0),
		_k3x(0.0), _k3y(0.0), _k3z(0.0),
		_k4x(0.0), _k4y(0.0), _k4z(0.0),
		_radius(radius),
		_state(state)
{}

SigAsiaDemo::MassList::MassList(
	float coeff_friction,
	float coeff_restitution,
	float plane_size,
	unsigned int threads) :
	_screen_width(1024),
	_screen_height(768),
	_masses_array(0),
	_masses_buffer(0),
	_computing(false),
	_changed(false),
	_coeff_friction(coeff_friction),
	_coeff_restitution(coeff_restitution),
	_device_masses(0),
	_plane_size(plane_size),
	_plane_array(0),
	_plane_buffer(0),
	_screen_array(0),
	_screen_pos_buffer(0),
	_screen_tex_buffer(0),
	_layer_0_ModelViewLocation(0),
	_layer_0_ProjectionLocation(0),
	_layer_0_vertex_shader(0),
	_layer_0_geometry_shader(0),
	_layer_0_fragment_shader(0),
	_layer_0_program(0),
	_layer_1_ModelViewLocation(0),
	_layer_1_ProjectionLocation(0),
	_layer_1_ColorTexLocation(0),
	_layer_1_vertex_shader(0),
	_layer_1_geometry_shader(0),
	_layer_1_fragment_shader(0),
	_layer_1_program(0),
	_plane_ModelViewLocation(0),
	_plane_ProjectionLocation(0),
	_plane_vertex_shader(0),
	_plane_fragment_shader(0),
	_plane_program(0),
	_screen_ColorTexLocation(0),
	_screen_vertex_shader(0),
	_screen_fragment_shader(0),
	_screen_program(0),
	_inv_rho(1.0),
	_image_width(1024),
	_image_height(768),
	_image_buffer(0),
	_image_color(0),
	_image_depth(0),
	_image2_buffer(0),
	_image2_color(0),
	_image2_depth(0),
	_threads(threads)
{
}

SigAsiaDemo::MassList::~MassList()
{
	if (_device_masses) {
		cudaThreadSynchronize();
		//std::cout << "Free masses." << std::endl;
		cudaFree(_device_masses);
		_device_masses = 0;
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

bool SigAsiaDemo::MassList::empty() const
{
	return _masses.empty();
}

size_t SigAsiaDemo::MassList::size() const
{
	return _masses.size();
}

void SigAsiaDemo::MassList::upload(bool force_copy)
{
	if (_computing) {
		// do nothing if computing
		std::cout << "Can not upload, computing." << std::endl;
		return;
	}

	if (force_copy) {
		// copy into GPU buffer
		//std::cout << "Copy masses into GPU buffer." << std::endl;
		cudaMemcpy(
			_device_masses,
			&_masses[0],
			_masses.size()*sizeof(Mass),
			cudaMemcpyHostToDevice);
	} else {
		if (_changed) {
			//std::cout << "Upload masses." << std::endl;
			_changed = false;
			if (_device_masses) {
				cudaThreadSynchronize();
				//std::cout << "Free masses." << std::endl;
				cudaFree(_device_masses);
				_device_masses = 0;
			}

			// allocate GPU buffer
			//std::cout << std::fixed << std::setprecision(8) \
			<< "Allocate GPU buffer of size " << \
			_masses.size()*sizeof(Mass)/1073741824.0 \
			<< " GB." << std::endl;
			cudaError_t result = cudaMalloc(
				(void**)&_device_masses,
				_masses.size()*sizeof(Mass));
			if (result != cudaSuccess) {
				std::cerr << "Error: CUDA failed to malloc memory." << std::endl;
				std::terminate();
			}

			// copy into GPU buffer
			//std::cout << "Copy masses into GPU buffer." << std::endl;
			cudaMemcpy(
				_device_masses,
				&_masses[0],
				_masses.size()*sizeof(Mass),
				cudaMemcpyHostToDevice);
		}
	}
}

void SigAsiaDemo::MassList::download()
{
	if (_computing && _changed) {
		std::cerr << "Error: Mass list changed while \
data was being used in GPU computations." << std::endl;
		std::terminate();
	} else {
		//std::cout << "Download masses" << std::endl;
		// copy into CPU buffer
		//std::cout << "Copy masses into CPU buffer." << std::endl;
		cudaMemcpy(
			&_masses[0],
			_device_masses,
			_masses.size()*sizeof(Mass),
			cudaMemcpyDeviceToHost);
	}
}

SigAsiaDemo::Mass *SigAsiaDemo::MassList::getMass(size_t index)
{
	if (_masses.empty()) {
		std::cerr << "Warning: getMass called on \
empty mass list." << std::endl;
		return 0;
	}
	if (index >= _masses.size()) {
		std::cerr << "Warning: getMass called on index \
out of bounds." << std::endl;
		return 0;
	}

	return &_masses[index];
}

SigAsiaDemo::Mass *SigAsiaDemo::MassList::getDeviceMasses()
{
	return _device_masses;
}

bool SigAsiaDemo::MassList::getChanged() const
{
	return _changed;
}

__global__ void deviceStartFrame(unsigned int N, SigAsiaDemo::Mass *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
	}
}

void SigAsiaDemo::MassList::startFrame()
{
	_computing = true;
	if (!_masses.empty()) {
		//std::cout << "Start frame (" \
		<< _masses.size() << ")." << std::endl;
		deviceStartFrame<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

void SigAsiaDemo::MassList::endFrame()
{
	_computing = false;
}

__global__ void deviceClearForces(
	unsigned int N,
	float fx,
	float fy,
	float fz,
	float gravity,
	SigAsiaDemo::Mass *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
		masses[tid]._fx = fx;
		masses[tid]._fy = fy + gravity * masses[tid]._mass;
		masses[tid]._fz = fz;
	}
}

void SigAsiaDemo::MassList::clearForces(
	float fx,	// force
	float fy,
	float fz,
	float gravity)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Clear forces and add gravity (" \
		<< _masses.size() << ")." << std::endl;
		deviceClearForces<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			_masses.size(),
			fx, fy, fz,
			gravity,
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK1(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N, SigAsiaDemo::Mass *masses,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k1
		masses[tid]._k1x = masses[tid]._vx + ax * dt;
		masses[tid]._k1y = masses[tid]._vy + ay * dt;
		masses[tid]._k1z = masses[tid]._vz + az * dt;

		// set temporary velocity for k2
		masses[tid]._tvx = masses[tid]._k1x;
		masses[tid]._tvy = masses[tid]._k1y;
		masses[tid]._tvz = masses[tid]._k1z;
		// set temporary position for k2
		masses[tid]._tx = masses[tid]._x + masses[tid]._tvx * dt * 0.5f;
		masses[tid]._ty = masses[tid]._y + masses[tid]._tvy * dt * 0.5f;
		masses[tid]._tz = masses[tid]._z + masses[tid]._tvz * dt * 0.5f;

		if (ground_collision && masses[tid]._ty < masses[tid]._radius) {
			masses[tid]._ty = masses[tid]._radius;
			// no slip condition
			masses[tid]._tvx =  masses[tid]._tvx * coeff_friction;
			masses[tid]._tvy = -masses[tid]._tvy * coeff_restitution;
			masses[tid]._tvz =  masses[tid]._tvz * coeff_friction;
		}
	}
}

void SigAsiaDemo::MassList::evaluateK1(
	float dt,
	bool ground_collision)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K1 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK1<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses,
			ground_collision);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK2(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N, SigAsiaDemo::Mass *masses,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k2
		masses[tid]._k2x = masses[tid]._vx + ax * dt;
		masses[tid]._k2y = masses[tid]._vy + ay * dt;
		masses[tid]._k2z = masses[tid]._vz + az * dt;

		// set temporary velocity for k3
		masses[tid]._tvx = masses[tid]._k2x;
		masses[tid]._tvy = masses[tid]._k2y;
		masses[tid]._tvz = masses[tid]._k2z;
		// set temporary position for k3
		masses[tid]._tx = masses[tid]._x + masses[tid]._tvx * dt * 0.5f;
		masses[tid]._ty = masses[tid]._y + masses[tid]._tvy * dt * 0.5f;
		masses[tid]._tz = masses[tid]._z + masses[tid]._tvz * dt * 0.5f;

		if (ground_collision && masses[tid]._ty < masses[tid]._radius) {
			masses[tid]._ty = masses[tid]._radius;
			// no slip condition
			masses[tid]._tvx =  masses[tid]._tvx * coeff_friction;
			masses[tid]._tvy = -masses[tid]._tvy * coeff_restitution;
			masses[tid]._tvz =  masses[tid]._tvz * coeff_friction;
		}
	}
}

void SigAsiaDemo::MassList::evaluateK2(
	float dt,
	bool ground_collision)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K2 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK2<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses,
			ground_collision);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK3(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N, SigAsiaDemo::Mass *masses,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k3
		masses[tid]._k3x = masses[tid]._vx +  ax * dt;
		masses[tid]._k3y = masses[tid]._vy +  ay * dt;
		masses[tid]._k3z = masses[tid]._vz +  az * dt;

		// set temporary velocity for k4
		masses[tid]._tvx = masses[tid]._k3x;
		masses[tid]._tvy = masses[tid]._k3y;
		masses[tid]._tvz = masses[tid]._k3z;
		// set temporary position for k4
		masses[tid]._tx = masses[tid]._x + masses[tid]._tvx * dt;
		masses[tid]._ty = masses[tid]._y + masses[tid]._tvy * dt;
		masses[tid]._tz = masses[tid]._z + masses[tid]._tvz * dt;

		if (ground_collision && masses[tid]._ty < masses[tid]._radius) {
			masses[tid]._ty = masses[tid]._radius;
			// no slip condition
			masses[tid]._tvx =  masses[tid]._tvx * coeff_friction;
			masses[tid]._tvy = -masses[tid]._tvy * coeff_restitution;
			masses[tid]._tvz =  masses[tid]._tvz * coeff_friction;
		}
	}
}

void SigAsiaDemo::MassList::evaluateK3(
	float dt,
	bool ground_collision)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K3 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK3<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK4(
	float dt,
	unsigned int N, SigAsiaDemo::Mass *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses[tid]._mass;
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses[tid]._fx * inv_mass;
		float ay = masses[tid]._fy * inv_mass;
		float az = masses[tid]._fz * inv_mass;

		// evaluate k4
		masses[tid]._k4x = masses[tid]._vx + ax * dt;
		masses[tid]._k4y = masses[tid]._vy + ay * dt;
		masses[tid]._k4z = masses[tid]._vz + az * dt;
	}
}

void SigAsiaDemo::MassList::evaluateK4(
	float dt,
	bool ground_collision)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K4 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK4<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceUpdate(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N,
	SigAsiaDemo::Mass *masses,
	float *masses_buffer,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses[tid]._state == 0) {
			// update position
			masses[tid]._x += 0.166666666666667f * (
				masses[tid]._k1x +
				masses[tid]._k2x*2.0f +
				masses[tid]._k3x*2.0f +
				masses[tid]._k4x);
			masses[tid]._y += 0.166666666666667f * (
				masses[tid]._k1y +
				masses[tid]._k2y*2.0f +
				masses[tid]._k3y*2.0f +
				masses[tid]._k4y);
			masses[tid]._z += 0.166666666666667f * (
				masses[tid]._k1z +
				masses[tid]._k2z*2.0f +
				masses[tid]._k3z*2.0f +
				masses[tid]._k4z);

			// update velocity
			// NOTE: (_tx, _ty, _tz) contains previous positions
			// using backward difference
			masses[tid]._vx = (masses[tid]._x - masses[tid]._tx);
			masses[tid]._vy = (masses[tid]._y - masses[tid]._ty);
			masses[tid]._vz = (masses[tid]._z - masses[tid]._tz);

			// set temporary position to current position and velocity
			masses[tid]._tx = masses[tid]._x;
			masses[tid]._ty = masses[tid]._y;
			masses[tid]._tz = masses[tid]._z;
			masses[tid]._tvx = masses[tid]._vx;
			masses[tid]._tvy = masses[tid]._vy;
			masses[tid]._tvz = masses[tid]._vz;

			// enforce ground collision
			if (ground_collision && masses[tid]._y < masses[tid]._radius) {
				masses[tid]._y = masses[tid]._radius;
				masses[tid]._ty = masses[tid]._radius;

				// no slip condition
				masses[tid]._vx =  masses[tid]._vx * coeff_friction;
				masses[tid]._vy = -masses[tid]._vy * coeff_restitution;
				masses[tid]._vz =  masses[tid]._vz * coeff_friction;
				masses[tid]._tvx =  masses[tid]._tvx * coeff_friction;
				masses[tid]._tvy = -masses[tid]._tvy * coeff_restitution;
				masses[tid]._tvz =  masses[tid]._tvz * coeff_friction;
			}
		}
		
		// copy into CUDA buffer
		masses_buffer[tid*4]   = masses[tid]._x;
		masses_buffer[tid*4+1] = masses[tid]._y;
		masses_buffer[tid*4+2] = masses[tid]._z;
		masses_buffer[tid*4+3] = masses[tid]._radius;
	}
}

void SigAsiaDemo::MassList::update(
	float dt,
	bool ground_collision)
{
	if (_changed && _masses_buffer != 0) {
		// unregister GL buffer
		cudaGraphicsUnregisterResource(_cuda_masses_resource);
		_cuda_masses_resource = 0;
		// clear GL buffer
		glDeleteBuffers(1, &_masses_buffer);
		_masses_buffer = 0;
	}
	if (_masses_buffer == 0) {
		// generate GL buffer
		glGenBuffers(1, &_masses_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, _masses_buffer);
		// allocate space for (position, radius);
		glBufferData(
			GL_ARRAY_BUFFER,
			_masses.size()*4*sizeof(float),
			NULL,
			GL_DYNAMIC_DRAW);
		// register GL buffer
		cudaGraphicsGLRegisterBuffer(
			&_cuda_masses_resource,
			_masses_buffer,
			cudaGraphicsMapFlagsNone);
		if (_cuda_masses_resource == 0) {
			std::cerr << "Error: Failed to register GL buffer." << std::endl;
			return;
		}
	}
	if (_plane_buffer == 0) {
		std::cout << "Generate ground plane buffer." << std::endl;
		glGenBuffers(1, &_plane_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, _plane_buffer);
		// allocate space for position
		float _plane_data[] = 
			{
				_plane_size, 0.0, _plane_size, 1.0,
				-_plane_size, 0.0, _plane_size, 1.0,
				-_plane_size, 0.0, -_plane_size, 1.0,
				_plane_size, 0.0, -_plane_size, 1.0
			};
		glBufferData(
			GL_ARRAY_BUFFER,
			16*sizeof(float),
			_plane_data,
			GL_DYNAMIC_DRAW);
	}
	if (_screen_pos_buffer == 0) {
		std::cout << "Generate screen quad position buffer." << std::endl;
		glGenBuffers(1, &_screen_pos_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, _screen_pos_buffer);
		// allocate space for position
		float _screen_data[] = 
			{
				1.0, 1.0, 0.0, 1.0,
				-1.0, 1.0, 0.0, 1.0,
				-1.0, -1.0, 0.0, 1.0,
				1.0, -1.0, 0.0, 1.0
			};
		glBufferData(
			GL_ARRAY_BUFFER,
			16*sizeof(float),
			_screen_data,
			GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	if (_screen_tex_buffer == 0) {
		std::cout << "Generate screen texture coordinates buffer." << std::endl;
		glGenBuffers(1, &_screen_tex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, _screen_tex_buffer);
		// allocate space for uvs
		float _screen_data[] = 
			{
				1.0, 1.0,
				0.0, 1.0,
				0.0, 0.0,
				1.0, 0.0
			};
		glBufferData(
			GL_ARRAY_BUFFER,
			8*sizeof(float),
			_screen_data,
			GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// generate arrays
	if (_masses_buffer != 0) {
		if (_masses_array == 0)
			glGenVertexArrays(1, &_masses_array);
		glBindVertexArray(_masses_array);

		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, _masses_buffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindVertexArray(0);
	}

	if (_plane_buffer != 0) {
		//std::cout << "Bind plane array." << std::endl;
		if (_plane_array == 0) {
			std::cout << "Generate vertex arrays." << std::endl;
			glGenVertexArrays(1, &_plane_array);
		}
		glBindVertexArray(_plane_array);

		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, _plane_buffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindVertexArray(0);
	}

	if (_screen_pos_buffer != 0 && _screen_tex_buffer != 0) {
		//std::cout << "Bind screen array." << std::endl;
		if (_screen_array == 0) {
			std::cout << "Generate vertex arrays." << std::endl;
			glGenVertexArrays(1, &_screen_array);
		}
		glBindVertexArray(_screen_array);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, _screen_pos_buffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindBuffer(GL_ARRAY_BUFFER, _screen_tex_buffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glBindVertexArray(0);
	}

	// map CUDA resource
	size_t buffer_size = 0;
	float *masses_buffer = 0;
	cudaGraphicsMapResources(1, &_cuda_masses_resource, NULL);
	cudaGraphicsResourceGetMappedPointer(
		(void**)&masses_buffer,
		&buffer_size,
		_cuda_masses_resource);
		

	// update positions and upload to GL
	if (_computing && !_masses.empty()) {
		//std::cout << "Update masses (" << _masses.size() << ")." \
		<< std::endl;
		deviceUpdate<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses,
			masses_buffer,
			ground_collision);
		cudaThreadSynchronize();
	}

	// unmap CUDA resource
	cudaGraphicsUnmapResources(1, &_cuda_masses_resource, NULL);

	// unbind buffer
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool verifyCompilation(unsigned int shader, const char *text, const char *type)
{
	GLint result = 0;
	glGetShaderiv(
		shader,
		GL_COMPILE_STATUS,
		&result);
	if (result == GL_FALSE) {
		std::cerr << "Error: Failed to compile " << type \
		<< " shader." << std::endl;
		GLint length = 0;
		glGetShaderiv(
			shader,
			GL_INFO_LOG_LENGTH,
			&length);
		if (length > 0) {
			GLint length_written = 0;
			char *log = new char[length];
			glGetShaderInfoLog(
				shader,
				length,
				&length_written,
				log);
			std::cerr << "Shader: " << std::endl;
			std::cerr << text << std::endl;
			std::cerr << "Log:" << std::endl;
			std::cerr << log << std::endl;
			delete[] log;
		}
		return false;
	}
	return true;
}

bool verifyLinking(unsigned int program)
{
	GLint result = 0;
	glGetProgramiv(
		program,
		GL_LINK_STATUS,
		&result);
	if (result == GL_FALSE) {
		std::cerr << "Error: Failed to compile shader program." \
		<< std::endl;
		GLint length = 0;
		glGetProgramiv(
			program,
			GL_INFO_LOG_LENGTH,
			&length);
		if (length > 0) {
			GLint length_written = 0;
			char *log = new char[length];
			glGetProgramInfoLog(
				program,
				length,
				&length_written,
				log);
			std::cerr << "Log:" << std::endl;
			std::cerr << log << std::endl;
			delete[] log;
		}
		return false;
	}
	return true;
}

bool loadShader(
	const char *vs_file_name,
	const char *gs_file_name,
	const char *fs_file_name,
	GLuint *program,
	GLuint *vertex_shader,
	GLuint *geometry_shader,
	GLuint *fragment_shader)
{
	if (!program || !vertex_shader || !fragment_shader || 
		!vs_file_name || !fs_file_name)
		return false;

	// load shaders
	if (*program == 0) {
		// read and compile shaders
		*vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		if (*vertex_shader == 0) {
			std::cerr << "Warning: Failed to create vertex shader." \
			<< std::endl;
			return false;
		}
		std::ifstream vs_file(vs_file_name);
		std::string vs_string(
			(std::istreambuf_iterator<char>(vs_file)),
			std::istreambuf_iterator<char>());
		const char *vs_char = vs_string.c_str();
		glShaderSource(*vertex_shader, 1, &vs_char, NULL);
		glCompileShader(*vertex_shader);
		if (verifyCompilation(
				*vertex_shader,
				vs_string.c_str(),
				"vertex") == false)
			return false;

		if (geometry_shader && gs_file_name) {
			*geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
			if (*geometry_shader == 0) {
				std::cerr << "Warning: Failed to create geometry shader." \
				<< std::endl;
			}
			std::ifstream gs_file(gs_file_name);
			std::string gs_string(
				(std::istreambuf_iterator<char>(gs_file)),
				std::istreambuf_iterator<char>());
			const char *gs_char = gs_string.c_str();
			glShaderSource(*geometry_shader, 1, &gs_char, NULL);
			glCompileShader(*geometry_shader);
			if (verifyCompilation(
					*geometry_shader,
					gs_string.c_str(),
					"geometry") == false)
				return false;
		}

		*fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		if (*fragment_shader == 0) {
			std::cerr << "Error: Failed to create fragment shader." \
			<< std::endl;
			return false;
		}
		std::ifstream fs_file(fs_file_name);
		std::string fs_string(
			(std::istreambuf_iterator<char>(fs_file)),
			std::istreambuf_iterator<char>());
		const char *fs_char = fs_string.c_str();
		glShaderSource(*fragment_shader, 1, &fs_char, NULL);
		glCompileShader(*fragment_shader);
		if (verifyCompilation(
				*fragment_shader,
				fs_string.c_str(),
				"fragment") == false)
			return false;

		// create program
		*program = glCreateProgram();
		if (*program == 0) {
			std::cerr << "Error: Failed to create shader program." \
			<< std::endl;
			return false;
		}

		// attach shaders
		glAttachShader(*program, *vertex_shader);
		if (geometry_shader && *geometry_shader)
			glAttachShader(*program, *geometry_shader);
		glAttachShader(*program, *fragment_shader);

		// bind attributes
		//glBindAttribLocation(*program, 0, "position");
		//glBindAttribLocation(*program, 1, "uv");

		// link program
		glLinkProgram(*program);
		if (verifyLinking(*program) == false)
			return false;

		glUseProgram(0);
	}
	return true;
}

bool SigAsiaDemo::MassList::loadShaders()
{
	std::cout << "Load layer 0 shader" << std::endl;
	bool success = false;
	success = loadShader(
		"massVS.glsl",
		"massGS.glsl",
		"massFS.glsl",
		&_layer_0_program,
		&_layer_0_vertex_shader,
		&_layer_0_geometry_shader,
		&_layer_0_fragment_shader);
	if (!success)
		return false;

	glUseProgram(_layer_0_program);

	// get uniforms
	_layer_0_ModelViewLocation = glGetUniformLocation(
		_layer_0_program, "ModelView");
	if (_layer_0_ModelViewLocation == -1) {
		std::cerr << "Error: Failed to get ModelView location." \
			<< std::endl;
		return false;
	}

	_layer_0_ProjectionLocation = glGetUniformLocation(
		_layer_0_program, "Projection");
	if (_layer_0_ProjectionLocation == -1) {
		std::cerr << "Error: Failed to get Projection location." \
			<< std::endl;
		return false;
	}

	std::cout << "Load layer 1 shader" << std::endl;
	success = loadShader(
		"massVS.glsl",
		"massGS.glsl",
		"mass2FS.glsl",
		&_layer_1_program,
		&_layer_1_vertex_shader,
		&_layer_1_geometry_shader,
		&_layer_1_fragment_shader);
	if (!success)
		return false;

	glUseProgram(_layer_1_program);

	// get uniforms
	_layer_1_ModelViewLocation = glGetUniformLocation(
		_layer_1_program, "ModelView");
	if (_layer_1_ModelViewLocation == -1) {
		std::cerr << "Error: Failed to get ModelView location." \
			<< std::endl;
		return false;
	}

	_layer_1_ProjectionLocation = glGetUniformLocation(
		_layer_1_program, "Projection");
	if (_layer_1_ProjectionLocation == -1) {
		std::cerr << "Error: Failed to get Projection location." \
			<< std::endl;
		return false;
	}

	_layer_1_ColorTexLocation = glGetUniformLocation(
		_layer_1_program, "color_tex");
	if (_layer_1_ColorTexLocation == -1) {
		std::cerr << "Error: Failed to get Color Tex location." \
			<< std::endl;
		return false;
	}

	std::cout << "Load plane shader" << std::endl;
	success = loadShader(
		"planeVS.glsl",
		"",
		"planeFS.glsl",
		&_plane_program,
		&_plane_vertex_shader,
		0,
		&_plane_fragment_shader);
	if (!success)
		return false;

	glUseProgram(_plane_program);

	// get uniforms
	_plane_ModelViewLocation = glGetUniformLocation(
		_plane_program, "ModelView");
	if (_plane_ModelViewLocation == -1) {
		std::cerr << "Error: Failed to get ModelView location." \
			<< std::endl;
		return false;
	}

	_plane_ProjectionLocation = glGetUniformLocation(
		_plane_program, "Projection");
	if (_plane_ProjectionLocation == -1) {
		std::cerr << "Error: Failed to get Projection location." \
			<< std::endl;
		return false;
	}

	std::cout << "Load screen shader" << std::endl;
	success = loadShader(
		"screenVS.glsl",
		"",
		"screenFS.glsl",
		&_screen_program,
		&_screen_vertex_shader,
		0,
		&_screen_fragment_shader);
	if (!success)
		return false;

	glUseProgram(_screen_program);

	// get uniforms
	_screen_ColorTexLocation = glGetUniformLocation(
		_screen_program, "color_tex");
	if (_screen_ColorTexLocation == -1) {
		std::cerr << "Error: Failed to get Color Tex location." \
			<< std::endl;
		return false;
	}

	glUseProgram(0);

	std::cout << "Finished loading shaders" << std::endl;

	return true;
}

void SigAsiaDemo::MassList::clearBuffers()
{
	if (_image_color != 0) {
		glDeleteTextures(1, &_image_color);
		_image_color = 0;
	}
	if (_image_depth != 0) {
		glDeleteRenderbuffers(1, &_image_depth);
		_image_depth = 0;
	}
	if (_image_buffer != 0) {
		glDeleteFramebuffers(1, &_image_buffer);
		_image_buffer = 0;
	}

	if (_image2_color != 0) {
		glDeleteTextures(1, &_image2_color);
		_image2_color = 0;
	}
	if (_image2_depth != 0) {
		glDeleteRenderbuffers(1, &_image2_depth);
		_image2_depth = 0;
	}
	if (_image2_buffer != 0) {
		glDeleteFramebuffers(1, &_image2_buffer);
		_image2_buffer = 0;
	}
}

bool SigAsiaDemo::MassList::loadBuffers()
{
	if (!GLEW_ARB_texture_float) {
		std::cout << "Warning: No floating point texture support." << std::endl;
	}
	if (_image_buffer == 0) {
		// generate offscreen rendering buffer
		glGenFramebuffers(1, &_image_buffer);
		glBindFramebuffer(GL_FRAMEBUFFER, _image_buffer);

		glGenTextures(1, &_image_color);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _image_color);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _image_width, _image_height,
			0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
			GL_LINEAR);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, _image_color, 0);

		glGenRenderbuffers(1, &_image_depth);
		glBindRenderbuffer(GL_RENDERBUFFER, _image_depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
			_image_width, _image_height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, _image_depth);
		
		GLenum targets[] = {GL_COLOR_ATTACHMENT0};
		glDrawBuffers(1, targets);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	if (_image2_buffer == 0) {
		// generate offscreen rendering buffer
		glGenFramebuffers(1, &_image2_buffer);
		glBindFramebuffer(GL_FRAMEBUFFER, _image2_buffer);

		glGenTextures(1, &_image2_color);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, _image2_color);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, _image_width, _image_height,
			0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
			GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
			GL_LINEAR);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, _image2_color, 0);

		glGenRenderbuffers(1, &_image2_depth);
		glBindRenderbuffer(GL_RENDERBUFFER, _image2_depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
			_image_width, _image_height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, _image2_depth);
		
		GLenum targets[] = {GL_COLOR_ATTACHMENT0};
		glDrawBuffers(1, targets);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	return true;
}

void SigAsiaDemo::MassList::render(
	glm::mat4 ModelView,
	glm::mat4 Projection) const
{
	if (_layer_0_program == 0) {
		std::cerr << "Warning: _layer_0_program not set." \
		<< std::endl;
		return;
	}
	if (_layer_1_program == 0) {
		std::cerr << "Warning: _layer_1_program not set." \
		<< std::endl;
		return;
	}
	if (_plane_program == 0) {
		std::cerr << "Warning: _plane_program not set." \
		<< std::endl;
		return;
	}
	if (_screen_program == 0) {
		std::cerr << "Warning: _screen_program not set." \
		<< std::endl;
		return;
	}
	if (_layer_0_ModelViewLocation == -1) {
		std::cerr << "Warning: _layer_0_ModelViewLocation not set." \
		<< std::endl;
		return;
	}
	if (_layer_0_ProjectionLocation == -1) {
		std::cerr << "Warning: _layer_0_ProjectionLocation not set." \
		<< std::endl;
		return;
	}
	if (_layer_1_ModelViewLocation == -1) {
		std::cerr << "Warning: _layer_1_ModelViewLocation not set." \
		<< std::endl;
		return;
	}
	if (_layer_1_ProjectionLocation == -1) {
		std::cerr << "Warning: _layer_1_ProjectionLocation not set." \
		<< std::endl;
		return;
	}
	if (_layer_1_ColorTexLocation == -1) {
		std::cerr << "Warning: _layer_1_ColorTexLocation not set." \
		<< std::endl;
		return;
	}
	if (_plane_ModelViewLocation == -1) {
		std::cerr << "Warning: _plane_ModelViewLocation not set." \
		<< std::endl;
		return;
	}
	if (_plane_ProjectionLocation == -1) {
		std::cerr << "Warning: _plane_ProjectionLocation not set." \
		<< std::endl;
		return;
	}
	if (_screen_ColorTexLocation == -1) {
		std::cerr << "Warning: _screen_ColorTexLocation not set." \
		<< std::endl;
		return;
	}
	if (_masses_array == 0) {
		std::cerr << "Warning: _masses_array not set." \
		<< std::endl;
		return;
	}
	if (_plane_array == 0) {
		std::cerr << "Warning: _plane_array not set." \
		<< std::endl;
		return;
	}
	if (_screen_array == 0) {
		std::cerr << "Warning: _screen_array not set." \
		<< std::endl;
		return;
	}
	if (_image_color == 0) {
		std::cerr << "Warning: _image_color not set." \
		<< std::endl;
		return;
	}
	if (_image2_color == 0) {
		std::cerr << "Warning: _image2_color not set." \
		<< std::endl;
		return;
	}

	//==============================
	// set viewport for image passes
	//==============================

	glViewport(0, 0, _image_width, _image_height);

	//=============================================
	// first rendering pass, draw into image buffer
	//=============================================

	// bind frame buffer
	glBindFramebuffer(GL_FRAMEBUFFER, _image_buffer);

	// clear frame buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// bind layer 0 shader
	glUseProgram(_layer_0_program);

	// setup uniforms
	glUniformMatrix4fv(
		_layer_0_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_layer_0_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));

	glBindVertexArray(_masses_array);
	glDrawArrays(GL_POINTS, 0, _masses.size());
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(_plane_program);

	glUniformMatrix4fv(
		_plane_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_plane_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));

	glBindVertexArray(_plane_array);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// unbind frame buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	//===============================================
	// second rendering pass, draw into image2 buffer
	//===============================================

	// TODO: remove
	glViewport(0, 0, _screen_width, _screen_height);

	// set depth function

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _image_color);

	/*
	// bind frame buffer
	glBindFramebuffer(GL_FRAMEBUFFER, _image2_buffer);

	// clear frame buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	*/

	// bind layer 1 shader
	glUseProgram(_layer_1_program);

	// setup uniforms
	glUniform1i(_layer_1_ColorTexLocation, 0);
	glUniformMatrix4fv(
		_layer_1_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_layer_1_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));

	glBindVertexArray(_masses_array);
	glDrawArrays(GL_POINTS, 0, _masses.size());
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDepthFunc(GL_GREATER);

	glBindVertexArray(_masses_array);
	glDrawArrays(GL_POINTS, 0, _masses.size());
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// reset depth function
	glDepthFunc(GL_LESS);

	glUseProgram(_plane_program);

	glUniformMatrix4fv(
		_plane_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_plane_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));

	glBindVertexArray(_plane_array);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// unbind frame buffer
	/*
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	*/

	//===============================
	// set viewport for screen passes
	//===============================

	glViewport(0, 0, _screen_width, _screen_height);

	//==========================================
	// render a screen space quad with the image
	//==========================================

	// bind quad shader
	/*
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _image2_color);

	glUseProgram(_screen_program);

	glUniform1i(_screen_ColorTexLocation, 0);

	glBindVertexArray(_screen_array);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	*/

	// unbind shader
	glUseProgram(0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void SigAsiaDemo::MassList::resizeWindow(
	float near,
	float far,
	float fov,
	float view_dist,
	float spring_length,
	unsigned int width,
	unsigned int height)
{
	_screen_width = width;
	_screen_height = height;
	// we assume that |v-p| ~= view_dist
	float lambda = (far*near * view_dist)/(far - near) + 1.0;
	_inv_rho = (tan(fov*0.5)*lambda)/(spring_length*static_cast<float>(width));
	std::cout << "rho: " << 1.0f / _inv_rho << std::endl;
	std::cout << "inv rho: " << _inv_rho << std::endl;
	std::cout << "lambda: " << lambda << std::endl;
	GLuint image_width =
		static_cast<GLuint>(static_cast<float>(width) * _inv_rho);
	GLuint image_height =
		static_cast<GLuint>(static_cast<float>(height) * _inv_rho);
	// TODO: remove
	image_width = width;
	image_height = height;
	if (image_width != _image_width || image_height != _image_height) {
		_image_width = image_width;
		_image_height = image_height;
		std::cout << "New image dimensions: [" << _image_width << ", " \
			<< _image_height << "]" << std::endl;
		clearBuffers();
		loadBuffers();
	}
}
