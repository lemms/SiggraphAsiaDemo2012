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
				float x = 0.0,
				float y = 0.0,
				float z = 0.0,
				float fx = 0.0,
				float fy = 0.0,
				float fz = 0.0,
				int state = 0,
				float radius = 1.0);

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
			void upload(bool force_copy = false);
			void download();
			Mass *getMass(size_t index);
			// returns 0 if Mass is uploaded to the GPU
			Mass *getDeviceMasses();
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
			std::vector<Mass> _masses;
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
			Mass *_device_masses;

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
