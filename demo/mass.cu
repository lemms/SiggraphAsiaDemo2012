/*
Siggraph Asia 2012 Demo

Mass vector implementation.

This file is part of SigAsiaDemo2012.

SigAsiaDemo2012 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SigAsiaDemo2012 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SigAsiaDemo2012.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2012 Laurence Emms

*/

//#define MASS_DEBUG

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <iterator>
#include <cmath>
using namespace std;

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

SigAsiaDemo::Mass::Mass(
		float mass,
		float x, float y, float z,
		float vx, float vy, float vz,
		float fx, float fy, float fz,
		float radius,
		int state,
		float k1x, float k1y, float k1z,
		float k2x, float k2y, float k2z,
		float k3x, float k3y, float k3z,
		float k4x, float k4y, float k4z) :
		_mass(mass),
		_x(x), _y(y), _z(z),
		_vx(vx), _vy(vy), _vz(vz),
		_fx(fx), _fy(fy), _fz(fz),
		_tx(x), _ty(y), _tz(z),
		_tvx(vz), _tvy(vy), _tvz(vz),
		_k1x(k1x), _k1y(k1y), _k1z(k1z),
		_k2x(k2x), _k2y(k2y), _k2z(k2z),
		_k3x(k3x), _k3y(k3y), _k3z(k3z),
		_k4x(k4x), _k4y(k4y), _k4z(k4z),
		_radius(radius),
		_state(state)
{}

SigAsiaDemo::MassList::MassList(
	float coeff_friction,
	float coeff_restitution,
	float plane_size,
	unsigned int threads) :
	_masses_array(0),
	_masses_buffer(0),
	_computing(false),
	_changed(false),
	_coeff_friction(coeff_friction),
	_coeff_restitution(coeff_restitution),
	_device_masses_ptr(0),
	_point_ModelViewLocation(0),
	_point_ProjectionLocation(0),
	_point_vertex_shader(0),
	_point_geometry_shader(0),
	_point_fragment_shader(0),
	_point_program(0),
	_threads(threads)
{
}

SigAsiaDemo::MassList::~MassList()
{
	if (_device_masses_ptr) {
		cudaFree(_device_masses_ptr);
		_device_masses_ptr = 0;
	}
}

bool SigAsiaDemo::MassList::push(Mass mass)
{
	// enforce that no masses can be added
	// if the buffer is uploaded to the GPU
	if (!_computing) {
		_masses.push(mass);
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

void SigAsiaDemo::MassList::upload()
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::upload()" << endl;
#endif
	if (_computing) {
		// do nothing if computing
		cout << "Can not upload, computing." << endl;
		return;
	}

	if (_changed) {
		// upload masses
		_changed = false;
		if (!_device_masses.invalid()) {
#ifdef MASS_DEBUG
		cout << "Free device masses." << endl;
#endif

			// free masses
			_device_masses.free();
		}

#ifdef MASS_DEBUG
		cout << "Upload masses." << endl;
#endif
		_device_masses.upload(_masses);

		// allocate masses structure for device
		cudaError_t result = cudaSuccess;
		if (!_device_masses_ptr) {
			result = cudaMalloc(
					&_device_masses_ptr,
					sizeof(MassDeviceArrays));
			if (result != cudaSuccess) {
				std::cerr << "Error: CUDA failed to malloc memory for _device_masses_ptr." << std::endl;
				std::cerr << cudaGetErrorString(result) << std::endl;
				std::terminate();
			}
		}

		result = cudaMemcpy(
				_device_masses_ptr,
				&_device_masses,
				sizeof(MassDeviceArrays),
				cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			std::cerr << "Error: CUDA failed to upload array." << std::endl;
			std::cerr << cudaGetErrorString(result) << std::endl;
			std::terminate();
		}
	}
}

void SigAsiaDemo::MassList::download()
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::download()" << endl;
#endif
	if (_computing && _changed) {
		cerr << "Error: Mass list changed while \
data was being used in GPU computations." << endl;
		terminate();
	} else {
		// copy into CPU buffer
#ifdef MASS_DEBUG
		cout << "Download masses." << endl;
#endif
		_device_masses.download(_masses);
	}
}

SigAsiaDemo::Mass SigAsiaDemo::MassList::getMass(size_t index)
{
	if (_computing) {
		cerr << "Error: getMass called while \
computing." << endl;
		terminate();
	}
	if (_masses.empty()) {
		cerr << "Error: getMass called on \
empty mass list." << endl;
		terminate();
	}
	if (index >= _masses.size()) {
		cerr << "Error: getMass called on index \
out of bounds." << endl;
		terminate();
	}


	return Mass(	_masses._mass[index],
			_masses._x[index], _masses._y[index], _masses._z[index],
			_masses._vx[index], _masses._vy[index], _masses._vz[index],
			_masses._fx[index], _masses._fy[index], _masses._fz[index],
			_masses._radius[index],
			_masses._state[index],
			_masses._k1x[index], _masses._k1y[index], _masses._k1z[index],
			_masses._k2x[index], _masses._k2y[index], _masses._k2z[index],
			_masses._k3x[index], _masses._k3y[index], _masses._k3z[index],
			_masses._k4x[index], _masses._k4y[index], _masses._k4z[index]);
}

SigAsiaDemo::MassDeviceArrays *SigAsiaDemo::MassList::getDeviceMasses()
{
	return &_device_masses;
}
SigAsiaDemo::MassDeviceArrays *SigAsiaDemo::MassList::getDeviceMassesPtr()
{
	return _device_masses_ptr;
}

bool SigAsiaDemo::MassList::getChanged() const
{
	return _changed;
}

__global__ void deviceStartFrame(unsigned int N, SigAsiaDemo::MassDeviceArrays *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 1)
			return;
	}
}

void SigAsiaDemo::MassList::startFrame()
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::startFrame()" << endl;
#endif
	if (!_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		_computing = true;
		deviceStartFrame<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			_masses.size(),
			_device_masses_ptr);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceStartFrame() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}
}

void SigAsiaDemo::MassList::endFrame()
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::endFrame()" << endl;
#endif
	_computing = false;
}

__global__ void deviceClearForces(
	unsigned int N,
	float fx,
	float fy,
	float fz,
	float gravity,
	SigAsiaDemo::MassDeviceArrays *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 1)
			return;
		masses->_fx[tid] = fx;
		masses->_fy[tid] = fy + gravity * masses->_mass[tid];
		masses->_fz[tid] = fz;
	}
}

void SigAsiaDemo::MassList::clearForces(
	float fx,	// force
	float fy,
	float fz,
	float gravity)
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::clearForces()" << endl;
#endif
	if (_computing && !_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		deviceClearForces<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			_masses.size(),
			fx, fy, fz,
			gravity,
			_device_masses_ptr);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceClearForces() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}
}

__global__ void deviceEvaluateK1(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N,
	SigAsiaDemo::MassDeviceArrays *masses,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses->_mass[tid];
		float ax = masses->_fx[tid] * inv_mass;
		float ay = masses->_fy[tid] * inv_mass;
		float az = masses->_fz[tid] * inv_mass;

		// evaluate k1
		masses->_k1x[tid] = masses->_vx[tid] + ax * dt;
		masses->_k1y[tid] = masses->_vy[tid] + ay * dt;
		masses->_k1z[tid] = masses->_vz[tid] + az * dt;

		// set temporary velocity for k2
		masses->_tvx[tid] = masses->_k1x[tid];
		masses->_tvy[tid] = masses->_k1y[tid];
		masses->_tvz[tid] = masses->_k1z[tid];
		// set temporary position for k2
		masses->_tx[tid] = masses->_x[tid] + masses->_tvx[tid] * dt * 0.5f;
		masses->_ty[tid] = masses->_y[tid] + masses->_tvy[tid] * dt * 0.5f;
		masses->_tz[tid] = masses->_z[tid] + masses->_tvz[tid] * dt * 0.5f;

		if (ground_collision && masses->_ty[tid] < masses->_radius[tid]) {
			masses->_ty[tid] = masses->_radius[tid];
			// no slip condition
			masses->_tvx[tid] =  masses->_tvx[tid] * coeff_friction;
			masses->_tvy[tid] = -masses->_tvy[tid] * coeff_restitution;
			masses->_tvz[tid] =  masses->_tvz[tid] * coeff_friction;
		}
	}
}

void SigAsiaDemo::MassList::evaluateK1(
	float dt,
	bool ground_collision)
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::evaluateK1()" << endl;
#endif
	if (_computing && !_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		deviceEvaluateK1<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses_ptr,
			ground_collision);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceEvaluateK1() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}
}

__global__ void deviceEvaluateK2(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N,
	SigAsiaDemo::MassDeviceArrays *masses,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses->_mass[tid];
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses->_fx[tid] * inv_mass;
		float ay = masses->_fy[tid] * inv_mass;
		float az = masses->_fz[tid] * inv_mass;

		// evaluate k2
		masses->_k2x[tid] = masses->_vx[tid] + ax * dt;
		masses->_k2y[tid] = masses->_vy[tid] + ay * dt;
		masses->_k2z[tid] = masses->_vz[tid] + az * dt;

		// set temporary velocity for k3
		masses->_tvx[tid] = masses->_k2x[tid];
		masses->_tvy[tid] = masses->_k2y[tid];
		masses->_tvz[tid] = masses->_k2z[tid];
		// set temporary position for k3
		masses->_tx[tid] = masses->_x[tid] + masses->_tvx[tid] * dt * 0.5f;
		masses->_ty[tid] = masses->_y[tid] + masses->_tvy[tid] * dt * 0.5f;
		masses->_tz[tid] = masses->_z[tid] + masses->_tvz[tid] * dt * 0.5f;

		if (ground_collision && masses->_ty[tid] < masses->_radius[tid]) {
			masses->_ty[tid] = masses->_radius[tid];
			// no slip condition
			masses->_tvx[tid] =  masses->_tvx[tid] * coeff_friction;
			masses->_tvy[tid] = -masses->_tvy[tid] * coeff_restitution;
			masses->_tvz[tid] =  masses->_tvz[tid] * coeff_friction;
		}
	}
}

void SigAsiaDemo::MassList::evaluateK2(
	float dt,
	bool ground_collision)
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::evaluateK2()" << endl;
#endif
	if (_computing && !_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		deviceEvaluateK2<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses_ptr,
			ground_collision);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceEvaluateK2() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}
}

__global__ void deviceEvaluateK3(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N,
	SigAsiaDemo::MassDeviceArrays *masses,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses->_mass[tid];
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses->_fx[tid] * inv_mass;
		float ay = masses->_fy[tid] * inv_mass;
		float az = masses->_fz[tid] * inv_mass;

		// evaluate k3
		masses->_k3x[tid] = masses->_vx[tid] +  ax * dt;
		masses->_k3y[tid] = masses->_vy[tid] +  ay * dt;
		masses->_k3z[tid] = masses->_vz[tid] +  az * dt;

		// set temporary velocity for k4
		masses->_tvx[tid] = masses->_k3x[tid];
		masses->_tvy[tid] = masses->_k3y[tid];
		masses->_tvz[tid] = masses->_k3z[tid];
		// set temporary position for k4
		masses->_tx[tid] = masses->_x[tid] + masses->_tvx[tid] * dt;
		masses->_ty[tid] = masses->_y[tid] + masses->_tvy[tid] * dt;
		masses->_tz[tid] = masses->_z[tid] + masses->_tvz[tid] * dt;

		if (ground_collision && masses->_ty[tid] < masses->_radius[tid]) {
			masses->_ty[tid] = masses->_radius[tid];
			// no slip condition
			masses->_tvx[tid] =  masses->_tvx[tid] * coeff_friction;
			masses->_tvy[tid] = -masses->_tvy[tid] * coeff_restitution;
			masses->_tvz[tid] =  masses->_tvz[tid] * coeff_friction;
		}
	}
}

void SigAsiaDemo::MassList::evaluateK3(
	float dt,
	bool ground_collision)
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::evaluateK3()" << endl;
#endif
	if (_computing && !_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		deviceEvaluateK3<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses_ptr);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceEvaluateK3() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}
}

__global__ void deviceEvaluateK4(
	float dt,
	unsigned int N,
	SigAsiaDemo::MassDeviceArrays *masses)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 1)
			return;
		// update accelerations
		float inv_mass = 1.0f / masses->_mass[tid];
		// forces have been calculated for (t + dt/2, y + k1/2)
		float ax = masses->_fx[tid] * inv_mass;
		float ay = masses->_fy[tid] * inv_mass;
		float az = masses->_fz[tid] * inv_mass;

		// evaluate k4
		masses->_k4x[tid] = masses->_vx[tid] + ax * dt;
		masses->_k4y[tid] = masses->_vy[tid] + ay * dt;
		masses->_k4z[tid] = masses->_vz[tid] + az * dt;
	}
}

void SigAsiaDemo::MassList::evaluateK4(
	float dt,
	bool ground_collision)
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::evaluateK4()" << endl;
#endif
	if (_computing && !_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		deviceEvaluateK4<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_masses.size(),
			_device_masses_ptr);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceEvaluateK4() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}
}

__global__ void deviceUpdate(
	float dt,
	float coeff_friction,
	float coeff_restitution,
	unsigned int N,
	SigAsiaDemo::MassDeviceArrays *masses,
	float *masses_buffer,
	bool ground_collision = true)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		if (masses->_state[tid] == 0) {
			// update position
			masses->_x[tid] += 0.166666666666667f * (
				masses->_k1x[tid] +
				masses->_k2x[tid]*2.0f +
				masses->_k3x[tid]*2.0f +
				masses->_k4x[tid]);
			masses->_y[tid] += 0.166666666666667f * (
				masses->_k1y[tid] +
				masses->_k2y[tid]*2.0f +
				masses->_k3y[tid]*2.0f +
				masses->_k4y[tid]);
			masses->_z[tid] += 0.166666666666667f * (
				masses->_k1z[tid] +
				masses->_k2z[tid]*2.0f +
				masses->_k3z[tid]*2.0f +
				masses->_k4z[tid]);

			// update velocity
			// NOTE: (_tx, _ty, _tz) contains previous positions
			// using backward difference
			masses->_vx[tid] = (masses->_x[tid] - masses->_tx[tid]);
			masses->_vy[tid] = (masses->_y[tid] - masses->_ty[tid]);
			masses->_vz[tid] = (masses->_z[tid] - masses->_tz[tid]);

			// set temporary position to current position and velocity
			masses->_tx[tid] = masses->_x[tid];
			masses->_ty[tid] = masses->_y[tid];
			masses->_tz[tid] = masses->_z[tid];
			masses->_tvx[tid] = masses->_vx[tid];
			masses->_tvy[tid] = masses->_vy[tid];
			masses->_tvz[tid] = masses->_vz[tid];

			// enforce ground collision
			if (ground_collision && masses->_y[tid] < masses->_radius[tid]) {
				masses->_y[tid] = masses->_radius[tid];
				masses->_ty[tid] = masses->_radius[tid];

				// no slip condition
				masses->_vx[tid] =  masses->_vx[tid] * coeff_friction;
				masses->_vy[tid] = -masses->_vy[tid] * coeff_restitution;
				masses->_vz[tid] =  masses->_vz[tid] * coeff_friction;
				masses->_tvx[tid] =  masses->_tvx[tid] * coeff_friction;
				masses->_tvy[tid] = -masses->_tvy[tid] * coeff_restitution;
				masses->_tvz[tid] =  masses->_tvz[tid] * coeff_friction;
			}
		}
		
		// copy into CUDA buffer
		masses_buffer[tid*4]   = masses->_x[tid];
		masses_buffer[tid*4+1] = masses->_y[tid];
		masses_buffer[tid*4+2] = masses->_z[tid];
		masses_buffer[tid*4+3] = masses->_radius[tid];
	}
}

void SigAsiaDemo::MassList::update(
	float dt,
	bool ground_collision)
{
#ifdef MASS_DEBUG
	cout << "SigAsiaDemo::MassList::update()" << endl;
#endif
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
			cerr << "Error: Failed to register GL buffer." << endl;
			return;
		}
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


	// map CUDA resource
	size_t buffer_size = 0;
	float *masses_buffer = 0;
	cudaGraphicsMapResources(1, &_cuda_masses_resource, NULL);
	cudaGraphicsResourceGetMappedPointer(
		(void**)&masses_buffer,
		&buffer_size,
		_cuda_masses_resource);
		

	// update positions and upload to GL
	if (_computing && !_masses.empty() && !_device_masses.invalid() && _device_masses_ptr) {
		deviceUpdate<<<(_masses.size()+_threads-1)/_threads, _threads>>>(
			dt,
			_coeff_friction,
			_coeff_restitution,
			_masses.size(),
			_device_masses_ptr,
			masses_buffer,
			ground_collision);
		cudaThreadSynchronize();
		cudaError_t result = cudaGetLastError();
		if (result != cudaSuccess) {
			cerr << "Error: deviceUpdate() failed with error: " << cudaGetErrorString(result) << endl;
			terminate();
		}
	}

	// unmap CUDA resource
	cudaGraphicsUnmapResources(1, &_cuda_masses_resource, NULL);

	// unbind buffer
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool SigAsiaDemo::verifyCompilation(unsigned int shader, const char *text, const char *type)
{
	GLint result = 0;
	glGetShaderiv(
		shader,
		GL_COMPILE_STATUS,
		&result);
	if (result == GL_FALSE) {
		cerr << "Error: Failed to compile " << type \
		<< " shader." << endl;
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
			cerr << "Shader: " << endl;
			cerr << text << endl;
			cerr << "Log:" << endl;
			cerr << log << endl;
			delete[] log;
		}
		return false;
	}
	return true;
}

bool SigAsiaDemo::verifyLinking(unsigned int program)
{
	GLint result = 0;
	glGetProgramiv(
		program,
		GL_LINK_STATUS,
		&result);
	if (result == GL_FALSE) {
		cerr << "Error: Failed to compile shader program." \
		<< endl;
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
			cerr << "Log:" << endl;
			cerr << log << endl;
			delete[] log;
		}
		return false;
	}
	return true;
}

bool SigAsiaDemo::loadShader(
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
			cerr << "Warning: Failed to create vertex shader." \
			<< endl;
			return false;
		}
		ifstream vs_file(vs_file_name);
		string vs_string(
			(istreambuf_iterator<char>(vs_file)),
			istreambuf_iterator<char>());
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
				cerr << "Warning: Failed to create geometry shader." \
				<< endl;
			}
			ifstream gs_file(gs_file_name);
			string gs_string(
				(istreambuf_iterator<char>(gs_file)),
				istreambuf_iterator<char>());
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
			cerr << "Error: Failed to create fragment shader." \
			<< endl;
			return false;
		}
		ifstream fs_file(fs_file_name);
		string fs_string(
			(istreambuf_iterator<char>(fs_file)),
			istreambuf_iterator<char>());
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
			cerr << "Error: Failed to create shader program." \
			<< endl;
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

	cout << "Load point shader" << endl;
	bool success = false;
	success = loadShader(
		"pointVS.glsl",
		"pointGS.glsl",
		"pointFS.glsl",
		&_point_program,
		&_point_vertex_shader,
		&_point_geometry_shader,
		&_point_fragment_shader);
	if (!success)
		return false;

	glUseProgram(_point_program);

	// get uniforms
	_point_ModelViewLocation = glGetUniformLocation(
		_point_program, "ModelView");
	if (_point_ModelViewLocation == -1) {
		cerr << "Error: Failed to get ModelView location." \
			<< endl;
		return false;
	}

	_point_ProjectionLocation = glGetUniformLocation(
		_point_program, "Projection");
	if (_point_ProjectionLocation == -1) {
		cerr << "Error: Failed to get Projection location." \
			<< endl;
		return false;
	}

	glUseProgram(0);

	cout << "Finished loading shaders" << endl;

	return true;
}

void SigAsiaDemo::MassList::render(
	glm::mat4 ModelView,
	glm::mat4 Projection) const
{
	if (_point_program == 0) {
		cerr << "Warning: _point_program not set." \
		<< endl;
		return;
	}

	if (_point_ModelViewLocation == -1) {
		cerr << "Warning: _point_ModelViewLocation not set." \
		<< endl;
		return;
	}
	if (_point_ProjectionLocation == -1) {
		cerr << "Warning: _point_ProjectionLocation not set." \
		<< endl;
		return;
	}

	// bind point shader
	glUseProgram(_point_program);

	// setup uniforms
	glUniformMatrix4fv(
		_point_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_point_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));

	glBindVertexArray(_masses_array);
	glDrawArrays(GL_POINTS, 0, _masses.size());
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// clear shader
	glUseProgram(0);
}
