/*
Siggraph Asia 2012 Demo

Mass vector interface.

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

//#define MASS_H_DEBUG

#include <vector>

namespace SigAsiaDemo {
	bool verifyCompilation(unsigned int shader, const char *text, const char *type);
	bool verifyLinking(unsigned int program);
	bool loadShader(
		const char *vs_file_name,
		const char *gs_file_name,
		const char *fs_file_name,
		GLuint *program,
		GLuint *vertex_shader,
		GLuint *geometry_shader,
		GLuint *fragment_shader);
	
	struct Mass {
		public:
			Mass(
				float mass = 0.0,
				float x = 0.0, float y = 0.0, float z = 0.0,
				float vx = 0.0, float vy = 0.0, float vz = 0.0,
				float fx = 0.0, float fy = 0.0, float fz = 0.0,
				float radius = 1.0,
				int state = 0,
				float k1x = 0.0, float k1y = 0.0, float k1z = 0.0,
				float k2x = 0.0, float k2y = 0.0, float k2z = 0.0,
				float k3x = 0.0, float k3y = 0.0, float k3z = 0.0,
				float k4x = 0.0, float k4y = 0.0, float k4z = 0.0
				);

			// data
			float _mass;
			// position
			float _x; float _y; float _z;
			// velocity
			float _vx; float _vy; float _vz;
			// force
			float _fx; float _fy; float _fz;
			// temporary position
			float _tx; float _ty; float _tz;
			// temporary velocity
			float _tvx; float _tvy; float _tvz;
			// RK4 components
			float _k1x; float _k1y; float _k1z;
			float _k2x; float _k2y; float _k2z;
			float _k3x; float _k3y; float _k3z;
			float _k4x; float _k4y; float _k4z;
			float _radius;
			// states:
			// 0 - unconstrained
			// 1 - held
			int _state;
	};

	struct MassHostArrays {
		void push(const Mass &mass)
		{
			_mass.push_back(mass._mass);
			_x.push_back(mass._x);
			_y.push_back(mass._y);
			_z.push_back(mass._z);
			_vx.push_back(mass._vx);
			_vy.push_back(mass._vy);
			_vz.push_back(mass._vz);
			_fx.push_back(mass._fx);
			_fy.push_back(mass._fy);
			_fz.push_back(mass._fz);
			_tx.push_back(mass._tx);
			_ty.push_back(mass._ty);
			_tz.push_back(mass._tz);
			_tvx.push_back(mass._tvx);
			_tvy.push_back(mass._tvy);
			_tvz.push_back(mass._tvz);
			_k1x.push_back(mass._k1x);
			_k1y.push_back(mass._k1y);
			_k1z.push_back(mass._k1z);
			_k2x.push_back(mass._k2x);
			_k2y.push_back(mass._k2y);
			_k2z.push_back(mass._k2z);
			_k3x.push_back(mass._k3x);
			_k3y.push_back(mass._k3y);
			_k3z.push_back(mass._k3z);
			_k4x.push_back(mass._k4x);
			_k4y.push_back(mass._k4y);
			_k4z.push_back(mass._k4z);
			_radius.push_back(mass._radius);
			_state.push_back(mass._state);
		}
		bool empty() const
		{
			return _mass.empty();
		}
		size_t size() const
		{
			return _mass.size();
		}
		std::vector<float> _mass;
		std::vector<float> _x; std::vector<float> _y; std::vector<float> _z;
		std::vector<float> _vx; std::vector<float> _vy; std::vector<float> _vz;
		std::vector<float> _fx; std::vector<float> _fy; std::vector<float> _fz;
		std::vector<float> _tx; std::vector<float> _ty; std::vector<float> _tz;
		std::vector<float> _tvx; std::vector<float> _tvy; std::vector<float> _tvz;
		std::vector<float> _k1x; std::vector<float> _k1y; std::vector<float> _k1z;
		std::vector<float> _k2x; std::vector<float> _k2y; std::vector<float> _k2z;
		std::vector<float> _k3x; std::vector<float> _k3y; std::vector<float> _k3z;
		std::vector<float> _k4x; std::vector<float> _k4y; std::vector<float> _k4z;
		std::vector<float> _radius;
		std::vector<int> _state;
	};
	struct MassDeviceArrays {
		MassDeviceArrays()
			: _invalid(true),
			_mass(0),
			_x(0), _y(0), _z(0),
			_vx(0), _vy(0), _vz(0),
			_fx(0), _fy(0), _fz(0),
			_tx(0), _ty(0), _tz(0),
			_tvx(0), _tvy(0), _tvz(0),
			_k1x(0), _k1y(0), _k1z(0),
			_k2x(0), _k2y(0), _k2z(0),
			_k3x(0), _k3y(0), _k3z(0),
			_radius(0),
			_state(0)
		{}
		~MassDeviceArrays()
		{
			free();
		}
		bool invalid() const
		{
			return _invalid;
		}
		void upload_array(void **device_array, const void *host_array, size_t data_size)
		{
			cudaError_t result = cudaSuccess;
			// allocate GPU buffer
			if (!(*device_array)) {
#ifdef MASS_H_DEBUG
				std::cout << "Allocate device array." << std::endl;
#endif
				result = cudaMalloc(
						device_array,
						data_size);
				if (result != cudaSuccess) {
					std::cerr << "Error: CUDA failed to malloc memory." << std::endl;
					std::cerr << cudaGetErrorString(result) << std::endl;
					std::terminate();
				}
			}

			// copy into GPU buffer
#ifdef MASS_H_DEBUG
			std::cout << "Upload device array." << std::endl;
#endif
			result = cudaMemcpy(
					*device_array,
					host_array,
					data_size,
					cudaMemcpyHostToDevice);
			if (result != cudaSuccess) {
				std::cerr << "Error: CUDA failed to upload array." << std::endl;
				std::cerr << cudaGetErrorString(result) << std::endl;
				std::terminate();
			}
		}
		void upload(MassHostArrays &host)
		{
			if (_invalid && host.size() > 0) {
				_invalid = false;

#ifdef MASS_H_DEBUG
				std::cout << "Upload arrays with size: " << host.size() << std::endl;
				std::cout << "Upload mass." << std::endl;
#endif
				upload_array((void**)&(_mass), (const void*)&(host._mass[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload position." << std::endl;
#endif
				upload_array((void**)&(_x), (const void*)&(host._x[0]), host.size()*sizeof(float));
				upload_array((void**)&(_y), (const void*)&(host._y[0]), host.size()*sizeof(float));
				upload_array((void**)&(_z), (const void*)&(host._z[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload velocity." << std::endl;
#endif
				upload_array((void**)&(_vx), (const void*)&(host._vx[0]), host.size()*sizeof(float));
				upload_array((void**)&(_vy), (const void*)&(host._vy[0]), host.size()*sizeof(float));
				upload_array((void**)&(_vz), (const void*)&(host._vz[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload force." << std::endl;
#endif
				upload_array((void**)&(_fx), (const void*)&(host._fx[0]), host.size()*sizeof(float));
				upload_array((void**)&(_fy), (const void*)&(host._fy[0]), host.size()*sizeof(float));
				upload_array((void**)&(_fz), (const void*)&(host._fz[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload temporary position." << std::endl;
#endif
				upload_array((void**)&(_tx), (const void*)&(host._tx[0]), host.size()*sizeof(float));
				upload_array((void**)&(_ty), (const void*)&(host._ty[0]), host.size()*sizeof(float));
				upload_array((void**)&(_tz), (const void*)&(host._tz[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload temporary velocity." << std::endl;
#endif
				upload_array((void**)&(_tvx), (const void*)&(host._tvx[0]), host.size()*sizeof(float));
				upload_array((void**)&(_tvy), (const void*)&(host._tvy[0]), host.size()*sizeof(float));
				upload_array((void**)&(_tvz), (const void*)&(host._tvz[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload k1." << std::endl;
#endif
				upload_array((void**)&(_k1x), (const void*)&(host._k1x[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k1y), (const void*)&(host._k1y[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k1z), (const void*)&(host._k1z[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload k2." << std::endl;
#endif
				upload_array((void**)&(_k2x), (const void*)&(host._k2x[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k2y), (const void*)&(host._k2y[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k2z), (const void*)&(host._k2z[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload k3." << std::endl;
#endif
				upload_array((void**)&(_k3x), (const void*)&(host._k3x[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k3y), (const void*)&(host._k3y[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k3z), (const void*)&(host._k3z[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload k4." << std::endl;
#endif
				upload_array((void**)&(_k4x), (const void*)&(host._k4x[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k4y), (const void*)&(host._k4y[0]), host.size()*sizeof(float));
				upload_array((void**)&(_k4z), (const void*)&(host._k4z[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload radius." << std::endl;
#endif
				upload_array((void**)&(_radius), (const void*)&(host._radius[0]), host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Upload state." << std::endl;
#endif
				upload_array((void**)&(_state), (const void*)&(host._state[0]), host.size()*sizeof(int));
#ifdef MASS_H_DEBUG
				if (!_mass) {
					std::cout << "Warning: _mass is not allocated." << std::endl;
					std::terminate();
				}
				if (!_x) {
					std::cout << "Warning: _x is not allocated." << std::endl;	
					std::terminate();
				}
				if (!_y) {
					std::cout << "Warning: _y is not allocated." << std::endl;
					std::terminate();
				}
				if (!_z) {
					std::cout << "Warning: _z is not allocated." << std::endl;
					std::terminate();
				}
				if (!_vx) {
					std::cout << "Warning: _vx is not allocated." << std::endl;
					std::terminate();
				}
				if (!_vy) {
					std::cout << "Warning: _vy is not allocated." << std::endl;
					std::terminate();
				}
				if (!_vz) {
					std::cout << "Warning: _vz is not allocated." << std::endl;
					std::terminate();
				}
				if (!_fx) {
					std::cout << "Warning: _fx is not allocated." << std::endl;
					std::terminate();
				}
				if (!_fy) {
					std::cout << "Warning: _fy is not allocated." << std::endl;
					std::terminate();
				}
				if (!_fz) {
					std::cout << "Warning: _fz is not allocated." << std::endl;
					std::terminate();
				}
				if (!_tx) {
					std::cout << "Warning: _tx is not allocated." << std::endl;
					std::terminate();
				}
				if (!_ty) {
					std::cout << "Warning: _ty is not allocated." << std::endl;
					std::terminate();
				}
				if (!_tz) {
					std::cout << "Warning: _tz is not allocated." << std::endl;
					std::terminate();
				}
				if (!_tvx) {
					std::cout << "Warning: _tvx is not allocated." << std::endl;
					std::terminate();
				}
				if (!_tvy) {
					std::cout << "Warning: _tvy is not allocated." << std::endl;
					std::terminate();
				}
				if (!_tvz) {
					std::cout << "Warning: _tvz is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k1x) {
					std::cout << "Warning: _k1x is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k1y) {
					std::cout << "Warning: _k1y is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k1z) {
					std::cout << "Warning: _k1z is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k2x) {
					std::cout << "Warning: _k2x is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k2y) {
					std::cout << "Warning: _k2y is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k2z) {
					std::cout << "Warning: _k2z is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k3x) {
					std::cout << "Warning: _k3x is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k3y) {
					std::cout << "Warning: _k3y is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k3z) {
					std::cout << "Warning: _k3z is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k4x) {
					std::cout << "Warning: _k4x is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k4y) {
					std::cout << "Warning: _k4y is not allocated." << std::endl;
					std::terminate();
				}
				if (!_k4z) {
					std::cout << "Warning: _k4z is not allocated." << std::endl;
					std::terminate();
				}
				if (!_radius) {
					std::cout << "Warning: _radius is not allocated." << std::endl;
					std::terminate();
				}
				if (!_state) {
					std::cout << "Warning: _state is not allocated." << std::endl;
					std::terminate();
				}
#endif
			}
		}
		void download_array(void *host_array, const void *device_array, size_t data_size)
		{
#ifdef MASS_H_DEBUG
			std::cout << "Download device array." << std::endl;
#endif
			cudaError_t result = cudaMemcpy(
					host_array,
					device_array,
					data_size,
					cudaMemcpyDeviceToHost);
			if (result != cudaSuccess) {
				std::cerr << "Error: CUDA failed to download array." << std::endl;
				std::cerr << cudaGetErrorString(result) << std::endl;
				std::terminate();
			}
		}
		void download(MassHostArrays &host)
		{
#ifdef MASS_H_DEBUG
			if (_invalid) {
				std::cout << "Warning: Array is invalid." << std::endl;
			}
			if (host.size() <= 0) {
				std::cout << "Warning: Host size is non-positive." << std::endl;
			}
#endif
			if (!_invalid && host.size() > 0) {
#ifdef MASS_H_DEBUG
				std::cout << "Download arrays with size: " << host.size() << std::endl;
				std::cout << "Download mass." << std::endl;
#endif
				download_array((void*)&(host._mass[0]), (const void*)_mass, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download position." << std::endl;
#endif
				download_array((void*)&(host._x[0]), (const void*)_x, host.size()*sizeof(float));
				download_array((void*)&(host._y[0]), (const void*)_y, host.size()*sizeof(float));
				download_array((void*)&(host._z[0]), (const void*)_z, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download velocity." << std::endl;
#endif
				download_array((void*)&(host._vx[0]), (const void*)_vx, host.size()*sizeof(float));
				download_array((void*)&(host._vy[0]), (const void*)_vy, host.size()*sizeof(float));
				download_array((void*)&(host._vz[0]), (const void*)_vz, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download force." << std::endl;
#endif
				download_array((void*)&(host._fx[0]), (const void*)_fx, host.size()*sizeof(float));
				download_array((void*)&(host._fy[0]), (const void*)_fy, host.size()*sizeof(float));
				download_array((void*)&(host._fz[0]), (const void*)_fz, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download temporary position." << std::endl;
#endif
				download_array((void*)&(host._tx[0]), (const void*)_tx, host.size()*sizeof(float));
				download_array((void*)&(host._ty[0]), (const void*)_ty, host.size()*sizeof(float));
				download_array((void*)&(host._tz[0]), (const void*)_tz, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download temporary velocity." << std::endl;
#endif
				download_array((void*)&(host._tvx[0]), (const void*)_tvx, host.size()*sizeof(float));
				download_array((void*)&(host._tvy[0]), (const void*)_tvy, host.size()*sizeof(float));
				download_array((void*)&(host._tvz[0]), (const void*)_tvz, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download k1." << std::endl;
#endif
				download_array((void*)&(host._k1x[0]), (const void*)_k1x, host.size()*sizeof(float));
				download_array((void*)&(host._k1y[0]), (const void*)_k1y, host.size()*sizeof(float));
				download_array((void*)&(host._k1z[0]), (const void*)_k1z, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download k2." << std::endl;
#endif
				download_array((void*)&(host._k2x[0]), (const void*)_k2x, host.size()*sizeof(float));
				download_array((void*)&(host._k2y[0]), (const void*)_k2y, host.size()*sizeof(float));
				download_array((void*)&(host._k2z[0]), (const void*)_k2z, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download k3." << std::endl;
#endif
				download_array((void*)&(host._k3x[0]), (const void*)_k3x, host.size()*sizeof(float));
				download_array((void*)&(host._k3y[0]), (const void*)_k3y, host.size()*sizeof(float));
				download_array((void*)&(host._k3z[0]), (const void*)_k3z, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download k4." << std::endl;
#endif
				download_array((void*)&(host._k4x[0]), (const void*)_k4x, host.size()*sizeof(float));
				download_array((void*)&(host._k4y[0]), (const void*)_k4y, host.size()*sizeof(float));
				download_array((void*)&(host._k4z[0]), (const void*)_k4z, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download radius." << std::endl;
#endif
				download_array((void*)&(host._radius[0]), (const void*)_radius, host.size()*sizeof(float));
#ifdef MASS_H_DEBUG
				std::cout << "Download state." << std::endl;
#endif
				download_array((void*)&(host._state[0]), (const void*)_state, host.size()*sizeof(int));
			}
		}
		void free()
		{
			cudaThreadSynchronize();
			_invalid = true;
			if (_mass) {cudaFree(_mass); _mass = 0;}
			if (_x) {cudaFree(_x); _x = 0;}
			if (_y) {cudaFree(_y); _y = 0;}
			if (_z) {cudaFree(_z); _z = 0;}
			if (_vx) {cudaFree(_vx); _vx = 0;}
			if (_vy) {cudaFree(_vy); _vy = 0;}
			if (_vz) {cudaFree(_vz); _vz = 0;}
			if (_fx) {cudaFree(_fx); _fx = 0;}
			if (_fy) {cudaFree(_fy); _fy = 0;}
			if (_fz) {cudaFree(_fz); _fz = 0;}
			if (_tx) {cudaFree(_tx); _tx = 0;}
			if (_ty) {cudaFree(_ty); _ty = 0;}
			if (_tz) {cudaFree(_tz); _tz = 0;}
			if (_tvx) {cudaFree(_tvx); _tvx = 0;}
			if (_tvy) {cudaFree(_tvy); _tvy = 0;}
			if (_tvz) {cudaFree(_tvz); _tvz = 0;}
			if (_k1x) {cudaFree(_k1x); _k1x = 0;}
			if (_k1y) {cudaFree(_k1y); _k1y = 0;}
			if (_k1z) {cudaFree(_k1z); _k1z = 0;}
			if (_k2x) {cudaFree(_k2x); _k2x = 0;}
			if (_k2y) {cudaFree(_k2y); _k2y = 0;}
			if (_k2z) {cudaFree(_k2z); _k2z = 0;}
			if (_k3x) {cudaFree(_k3x); _k3x = 0;}
			if (_k3y) {cudaFree(_k3y); _k3y = 0;}
			if (_k3z) {cudaFree(_k3z); _k3z = 0;}
			if (_radius) {cudaFree(_radius); _radius = 0;}
			if (_state) {cudaFree(_state); _state = 0;}
		}
		bool _invalid;
		float *_mass;
		float *_x; float *_y; float *_z;
		float *_vx; float *_vy; float *_vz;
		float *_fx; float *_fy; float *_fz;
		float *_tx; float *_ty; float *_tz;
		float *_tvx; float *_tvy; float *_tvz;
		float *_k1x; float *_k1y; float *_k1z;
		float *_k2x; float *_k2y; float *_k2z;
		float *_k3x; float *_k3y; float *_k3z;
		float *_k4x; float *_k4y; float *_k4z;
		float *_radius;
		int *_state;
	};

	class MassList {
		public:
			MassList(
				float coeff_friction = 0.2,
				float coeff_restitution = 0.2,
				float plane_size = 512.0,
				unsigned int threads = 1024);
			~MassList();
			bool push(Mass mass);
			bool empty() const;
			size_t size() const;
			void upload();
			void download();
			Mass getMass(size_t index);
			// returns 0 if Mass is uploaded to the GPU
			MassDeviceArrays *getDeviceMasses();
			MassDeviceArrays *getDeviceMassesPtr();
			bool getChanged() const;
			void startFrame();
			void endFrame();
			void clearForces(
				float fx = 0.0, float fy = 0.0, float fz = 0.0,
				float gravity = -9.8);
			void evaluateK1(
				float dt,
				bool ground_collision = true);
			void evaluateK2(
				float dt,
				bool ground_collision = true);
			void evaluateK3(
				float dt,
				bool ground_collision = true);
			void evaluateK4(
				float dt,
				bool ground_collision = true);
			void update(
				float dt,
				bool ground_collision = true);
			bool loadShaders();
			void render(glm::mat4 ModelView, glm::mat4 Projection) const;
		private:
			// vertex buffer object with (position, radius)
			GLuint _masses_array;
			GLuint _masses_buffer;
			cudaGraphicsResource *_cuda_masses_resource;
			// indicates that the GPU is currently
			// computing updates for the masses
			bool _computing;
			// indicates that the mass list has
			// changed
			bool _changed;
			float _coeff_friction;
			float _coeff_restitution;

			// arrays
			MassHostArrays _masses;
			MassDeviceArrays _device_masses;
			MassDeviceArrays *_device_masses_ptr;

			// shaders

			// point shader
			int _point_ModelViewLocation;
			int _point_ProjectionLocation;
			GLuint _point_vertex_shader;
			GLuint _point_geometry_shader;
			GLuint _point_fragment_shader;
			GLuint _point_program;

			// CUDA
			unsigned int _threads;
	};
}
