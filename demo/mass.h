/*
Siggraph Asia 2012 Demo

Mass vector interface.

Laurence Emms
*/

#include <vector>

namespace SigAsiaDemo {
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
				unsigned int threads = 128);
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
			void clearBuffers();
			bool loadBuffers();
			void render(glm::mat4 ModelView, glm::mat4 Projection) const;
			void resizeWindow(
				float near,
				float far,
				float fov,
				float view_dist,
				float spring_length,
				unsigned int width,
				unsigned int height);
		private:
			GLuint _screen_width;
			GLuint _screen_height;
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

			// ground plane
			float _plane_size;
			GLuint _plane_array;
			GLuint _plane_buffer;

			// screen_quad
			GLuint _screen_array;
			GLuint _screen_pos_buffer;
			GLuint _screen_tex_buffer;

			// shaders
			int _layer_0_ModelViewLocation;
			int _layer_0_ProjectionLocation;
			GLuint _layer_0_vertex_shader;
			GLuint _layer_0_geometry_shader;
			GLuint _layer_0_fragment_shader;
			GLuint _layer_0_program;

			int _plane_ModelViewLocation;
			int _plane_ProjectionLocation;
			GLuint _plane_vertex_shader;
			GLuint _plane_fragment_shader;
			GLuint _plane_program;

			int _screen_ColorTexLocation;
			GLuint _screen_vertex_shader;
			GLuint _screen_fragment_shader;
			GLuint _screen_program;

			// buffers
			float _inv_rho;
			GLuint _image_width;
			GLuint _image_height;
			GLuint _image_buffer;
			GLuint _image_color;
			GLuint _image_depth;

			// CUDA
			unsigned int _threads;
	};
}
