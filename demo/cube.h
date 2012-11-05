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
			float _radius;

		public:
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
		private:
			std::vector<Cube> _cubes;

			// collisions
			float _ks;
			float _kd;

			// CUDA
			unsigned int _threads;
	};
}
