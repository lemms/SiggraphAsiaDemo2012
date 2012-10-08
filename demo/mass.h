/*
Siggraph Asia 2012 Demo

Mass vector interface

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
				float vx = 0.0,
				float vy = 0.0,
				float vz = 0.0,
				float ax = 0.0,
				float ay = 0.0,
				float az = 0.0);

			// data
			float _x;
			float _y;
			float _z;
			float _vx;
			float _vy;
			float _vz;
			float _ax;
			float _ay;
			float _az;
			float _mass;
	};

	class MassList {
		public:
			MassList();
			~MassList();
			bool push(Mass mass);
			size_t size() const;
			void upload();
			void download();
			Mass *getMass(size_t index);
			// returns 0 if Mass is uploaded to the GPU
			Mass *getDeviceMasses();
			void clearAcceleration();
			void update(float dt);
		private:
			std::vector<Mass> _masses;
			bool _computing;
			// indicates that the GPU is currently
			// computing updates for the masses
			bool _changed;
			// indicates that the mass list has
			// changed
			Mass *_device_masses;
	};
}
