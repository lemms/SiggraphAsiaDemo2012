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
				float radius = 1.0);

			// data
			float _mass;
			// position
			float _x; float _y; float _z;
			// temporary position
			float _tx; float _ty; float _tz;
			// temporary velocity
			float _tvx; float _tvy; float _tvz;
			// forces
			float _fx; float _fy; float _fz;
			// RK4 components
			float _k1x; float _k1y; float _k1z;
			float _k2x; float _k2y; float _k2z;
			float _k3x; float _k3y; float _k3z;
			float _k4x; float _k4y; float _k4z;
			float _radius;
	};

	class MassList {
		public:
			MassList(float coeff_restitution = 0.2);
			~MassList();
			bool push(Mass mass);
			bool empty() const;
			size_t size() const;
			void upload();
			void download();
			Mass *getMass(size_t index);
			// returns 0 if Mass is uploaded to the GPU
			Mass *getDeviceMasses();
			bool getChanged() const;
			void startFrame();
			void clearForces();
			void evaluateK1(float dt);
			void evaluateK2(float dt);
			void evaluateK3(float dt);
			void evaluateK4(float dt);
			void update(float dt);
			void render() const;
		private:
			// vertex buffer object with (position, radius)
			std::vector<Mass> _masses;
			unsigned int _masses_buffer;
			cudaGraphicsResource *_cuda_masses_resource;
			// indicates that the GPU is currently
			// computing updates for the masses
			bool _computing;
			// indicates that the mass list has
			// changed
			bool _changed;
			float _coeff_restitution;
			Mass *_device_masses;
	};
}
