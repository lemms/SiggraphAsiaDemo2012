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
	_masses_array(0),
	_masses_buffer(0),
	_computing(false),
	_changed(false),
	_coeff_friction(coeff_friction),
	_coeff_restitution(coeff_restitution),
	_device_masses(0),
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
	if (_device_masses) {
		cudaThreadSynchronize();
		
		// free masses
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
		cout << "Can not upload, computing." << endl;
		return;
	}

	if (force_copy) {
		// copy into GPU buffer
		cudaMemcpy(
			_device_masses,
			&_masses[0],
			_masses.size()*sizeof(Mass),
			cudaMemcpyHostToDevice);
	} else {
		if (_changed) {
			// upload masses
			_changed = false;
			if (_device_masses) {
				cudaThreadSynchronize();

				// free masses
				cudaFree(_device_masses);
				_device_masses = 0;
			}

			// allocate GPU buffer
			cudaError_t result = cudaMalloc(
				(void**)&_device_masses,
				_masses.size()*sizeof(Mass));
			if (result != cudaSuccess) {
				cerr << "Error: CUDA failed to malloc memory." << endl;
				std::terminate();
			}

			// copy into GPU buffer
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
		cerr << "Error: Mass list changed while \
data was being used in GPU computations." << endl;
		std::terminate();
	} else {
		// copy into CPU buffer
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
		cerr << "Warning: getMass called on \
empty mass list." << endl;
		return 0;
	}
	if (index >= _masses.size()) {
		cerr << "Warning: getMass called on index \
out of bounds." << endl;
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
	if (_computing && !_masses.empty()) {
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
				cerr << "Warning: Failed to create geometry shader." \
				<< endl;
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
			cerr << "Error: Failed to create fragment shader." \
			<< endl;
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
