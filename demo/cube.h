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
		private:
			// indices
			unsigned int _start;
			unsigned int _end;

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
	};

	class CubeList {
		public:
			CubeList(
				unsigned int threads = 128);
			void push(Cube cube);
			bool empty() const;
			size_t size() const;
			void create(
				MassList &masses,
				SpringList &springs);
			void computeBounds();
		private:
			std::vector<Cube> _cubes;

			// CUDA
			unsigned int _threads;
	};
}
