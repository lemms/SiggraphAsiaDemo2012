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
	float coeff_restitution) :
	_masses_array(0),
	_masses_buffer(0),
	_computing(false),
	_changed(false),
	_coeff_restitution(coeff_restitution),
	_device_masses(0),
	_axes_array(0),
	_axes_buffer(0),
	_vertex_shader(0),
	_geometry_shader(0),
	_fragment_shader(0),
	_program(0),
	_ModelViewLocation(0),
	_ProjectionLocation(0)
{}

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

void SigAsiaDemo::MassList::upload()
{
	if (_computing) {
		// do nothing if computing
		return;
	}

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

	_computing = true;
}

void SigAsiaDemo::MassList::download()
{
	if (_changed) {
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
	_computing = false;
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
	int tid = blockIdx.x;
	if (tid < N) {
		if (masses[tid]._state == 1)
			return;
	}
}

void SigAsiaDemo::MassList::startFrame()
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Start frame (" \
		<< _masses.size() << ")." << std::endl;
		deviceStartFrame<<<_masses.size(), 1>>>(
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceClearForces(
	unsigned int N,
	float fx,
	float fy,
	float fz,
	float gravity,
	SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
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
		deviceClearForces<<<_masses.size(), 1>>>(
			_masses.size(),
			fx, fy, fz,
			gravity,
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK1(
	float dt, unsigned int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
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
	}
}

void SigAsiaDemo::MassList::evaluateK1(float dt)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K1 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK1<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK2(
	float dt, unsigned int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
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
	}
}

void SigAsiaDemo::MassList::evaluateK2(float dt)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K2 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK2<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK3(
	float dt, unsigned int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
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
	}
}

void SigAsiaDemo::MassList::evaluateK3(float dt)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K3 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK3<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceEvaluateK4(
	float dt, unsigned int N, SigAsiaDemo::Mass *masses)
{
	int tid = blockIdx.x;
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

void SigAsiaDemo::MassList::evaluateK4(float dt)
{
	if (_computing && !_masses.empty()) {
		//std::cout << "Evaluate K4 (" << _masses.size() << ")." << std::endl;
		deviceEvaluateK4<<<_masses.size(), 1>>>(
			dt,
			_masses.size(),
			_device_masses);
		cudaThreadSynchronize();
	}
}

__global__ void deviceUpdate(
	float dt,
	float coeff_restitution,
	unsigned int N,
	SigAsiaDemo::Mass *masses,
	float *masses_buffer,
	bool ground_collision = true)
{
	int tid = blockIdx.x;
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
			if (ground_collision && masses[tid]._y < 0.0f) {
				masses[tid]._y = 0.0f;
				masses[tid]._tvy = -masses[tid]._tvy * coeff_restitution;
			}
		}
		
		// copy into CUDA buffer
		masses_buffer[tid*4]   = masses[tid]._x;
		masses_buffer[tid*4+1] = masses[tid]._y;
		masses_buffer[tid*4+2] = masses[tid]._z;
		masses_buffer[tid*4+3] = masses[tid]._radius;
	}
}

void SigAsiaDemo::MassList::update(float dt)
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
	if (_axes_buffer == 0) {
		std::cout << "Generate axes_buffer." << std::endl;
		glGenBuffers(1, &_axes_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, _axes_buffer);
		// allocate space for (position, radius);
		float _axes_data[] = 
			{
				0.0, 0.0, 0.0, 0.1,
				1.0, 0.0, 0.0, 0.1,
				0.0, 1.0, 0.0, 0.1,
				0.0, 0.0, 1.0, 0.1
			};
		glBufferData(
			GL_ARRAY_BUFFER,
			16*sizeof(float),
			_axes_data,
			GL_DYNAMIC_DRAW);
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

	if (_axes_buffer != 0) {
		//std::cout << "Bind axes array." << std::endl;
		if (_axes_array == 0) {
			std::cout << "Generate vertex arrays." << std::endl;
			glGenVertexArrays(1, &_axes_array);
		}
		glBindVertexArray(_axes_array);

		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, _axes_buffer);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
		//glBindVertexArray(0);
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
		//std::cout << "Update masses (" << _masses.size() << ")." << std::endl;
		deviceUpdate<<<_masses.size(), 1>>>(
			dt,
			_coeff_restitution,
			_masses.size(),
			_device_masses,
			masses_buffer);
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

bool SigAsiaDemo::MassList::loadShaders()
{
	// load shaders
	if (_program == 0) {
		// read and compile shaders
		_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		if (_vertex_shader == 0) {
			std::cerr << "Error: Failed to create vertex shader." \
			<< std::endl;
			return false;
		}
		std::ifstream vs_file("massVS.glsl");
		std::string vs_string(
			(std::istreambuf_iterator<char>(vs_file)),
			std::istreambuf_iterator<char>());
		const char *vs_char = vs_string.c_str();
		glShaderSource(_vertex_shader, 1, &vs_char, NULL);
		glCompileShader(_vertex_shader);
		if (verifyCompilation(
				_vertex_shader,
				vs_string.c_str(),
				"vertex") == false)
			return false;

		_geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
		if (_geometry_shader == 0) {
			std::cerr << "Error: Failed to create geometry shader." \
			<< std::endl;
			return false;
		}
		std::ifstream gs_file("massGS.glsl");
		std::string gs_string(
			(std::istreambuf_iterator<char>(gs_file)),
			std::istreambuf_iterator<char>());
		const char *gs_char = gs_string.c_str();
		glShaderSource(_geometry_shader, 1, &gs_char, NULL);
		glCompileShader(_geometry_shader);
		if (verifyCompilation(
				_geometry_shader,
				gs_string.c_str(),
				"geometry") == false)
			return false;

		_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		if (_fragment_shader == 0) {
			std::cerr << "Error: Failed to create fragment shader." \
			<< std::endl;
			return false;
		}
		std::ifstream fs_file("massFS.glsl");
		std::string fs_string(
			(std::istreambuf_iterator<char>(fs_file)),
			std::istreambuf_iterator<char>());
		const char *fs_char = fs_string.c_str();
		glShaderSource(_fragment_shader, 1, &fs_char, NULL);
		glCompileShader(_fragment_shader);
		if (verifyCompilation(
				_fragment_shader,
				fs_string.c_str(),
				"fragment") == false)
			return false;

		// create program
		_program = glCreateProgram();
		if (_program == 0) {
			std::cerr << "Error: Failed to create shader program." \
			<< std::endl;
			return false;
		}

		// attach shaders
		glAttachShader(_program, _vertex_shader);
		glAttachShader(_program, _geometry_shader);
		glAttachShader(_program, _fragment_shader);

		// bind attributes
		glBindAttribLocation(_program, 0, "position");

		// link program
		glLinkProgram(_program);
		if (verifyLinking(_program) == false)
			return false;

		glUseProgram(_program);
		// get uniforms
		_ModelViewLocation = glGetUniformLocation(_program, "ModelView");
		if (_ModelViewLocation == -1) {
			std::cerr << "Error: Failed to get ModelView location." \
			<< std::endl;
			return false;
		}

		_ProjectionLocation = glGetUniformLocation(_program, "Projection");
		if (_ProjectionLocation == -1) {
			std::cerr << "Error: Failed to get Projection location." \
			<< std::endl;
			return false;
		}
		glUseProgram(0);
	}

	return true;
}

void SigAsiaDemo::MassList::render(
	glm::mat4 ModelView,
	glm::mat4 Projection) const
{
	if (_ModelViewLocation == -1) {
		std::cerr << "Warning: _ModelViewLocation not set." \
		<< std::endl;
		return;
	}
	if (_ProjectionLocation == -1) {
		std::cerr << "Warning: _ProjectionLocation not set." \
		<< std::endl;
		return;
	}
	if (_masses_array == 0) {
		std::cerr << "Warning: _masses_array not set." \
		<< std::endl;
		return;
	}
	if (_axes_array == 0) {
		std::cerr << "Warning: _axes_array not set." \
		<< std::endl;
		return;
	}

	// bind shader
	glUseProgram(_program);

	// setup uniforms
	glUniformMatrix4fv(
		_ModelViewLocation,
		1, GL_FALSE,
		glm::value_ptr(ModelView));
	glUniformMatrix4fv(
		_ProjectionLocation,
		1, GL_FALSE,
		glm::value_ptr(Projection));

	glBindVertexArray(_masses_array);
	glDrawArrays(GL_POINTS, 0, _masses.size());
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(_axes_array);
	glDrawArrays(GL_POINTS, 0, 4);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// unbind shader
	glUseProgram(0);
}
