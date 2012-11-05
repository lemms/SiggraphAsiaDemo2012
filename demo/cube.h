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
				size_t x, // multiple of 2
				size_t y,
				size_t z,
				float spacing,
				float mass,
				float radius);
			virtual ~Cube();
			virtual void create(
				float x,	// position
				float y,
				float z,
				MassList &masses,
				SpringList &springs);
		private:
			// indices
			unsigned int _start;
			unsigned int _end;

			// sizes
			int _half_x;
			int _half_y;
			int _half_z;

			float _spacing;

			// properties
			float _mass;
			float _radius;
	};
}
