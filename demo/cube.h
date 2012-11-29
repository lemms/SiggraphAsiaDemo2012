/*
Siggraph Asia 2012 Demo

Cube interface.

Laurence Emms
*/

#include <vector>

namespace SigAsiaDemo {
	class MassList;
	class SpringList;
	class Cube : public Creator {
		public:
			Cube(
				float x,	// position
				float y,
				float z,
				size_t size_x, // multiple of 2
				size_t size_y,
				size_t size_z,
				float spacing,
				float mass,
				float radius);
			virtual ~Cube();
			virtual void create(
				MassList &masses,
				SpringList &springs);

		public:
			// indices
			unsigned int _start;
			unsigned int _end;
			unsigned int _spring_start;
			unsigned int _spring_end;

			size_t _size_x;
			size_t _size_y;
			size_t _size_z;

		private:
			// position
			float _x;
			float _y;
			float _z;

			// sizes
			int _half_x;
			int _half_y;
			int _half_z;


			float _spacing;

			// properties
			float _mass;

		public:
			float _radius;

			// bounds
			float _min_x;
			float _min_y;
			float _min_z;
			float _max_x;
			float _max_y;
			float _max_z;
	};

	class CubeList {
		public:
			CubeList(
				float ks = 10000.0,
				float kd = 1000.0,
				unsigned int threads = 128);
			void setConstants(float ks, float kd);
			void push(Cube cube);
			bool empty() const;
			size_t size() const;
			void create(
				MassList &masses,
				SpringList &springs);
			Cube *getCube(size_t index);
			void computeBounds(
				MassList &masses);
			void collideCubes(
				float dt,
				MassList &masses);
			bool loadShaders();
			void render(
				glm::mat4 ModelView,
				glm::mat4 Projection,
				glm::mat3 Normal) const;
		private:
			std::vector<Cube> _cubes;
			std::vector<float> _tri_positions;
			std::vector<float> _tri_normals;
			std::vector<float> _tri_colors;

			// collisions
			float _ks;
			float _kd;

			// CUDA
			unsigned int _threads;

			// shader
			int _cube_ModelViewLocation;
			int _cube_ProjectionLocation;
			int _cube_NormalLocation;
			GLuint _cube_vertex_shader;
			GLuint _cube_fragment_shader;
			GLuint _cube_program;

			// buffers
			GLuint _cube_array;
			GLuint _cube_pos_buffer;
			GLuint _cube_norm_buffer;
			GLuint _cube_col_buffer;
	};
}
