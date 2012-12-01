/*
Siggraph Asia 2012 Demo

Spring vector interface.
Note that we are using simplified linear springs.

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

namespace SigAsiaDemo {
	class MassList;
	class CubeList;
	class Spring {
		public:
			Spring(
				MassList &masses,
				unsigned int mass0,
				unsigned int mass1);

			// data
			unsigned int _mass0; // mass 0 index
			unsigned int _mass1; // mass 1 index
			float _l0; // resting length
			// force applied to mass 0
			float _fx0; float _fy0; float _fz0;
			// force applied to mass 1
			float _fx1; float _fy1; float _fz1;
	};

	class SpringList {
		public:
			SpringList(
				float ks = 10000.0,
				float kd = 1000.0,
				unsigned int threads = 1024);
			~SpringList();
			void setConstants(float ks, float kd);
			bool push(Spring spring);
			bool empty() const;
			size_t size() const;
			void upload(MassList &masses, CubeList &cubes);
			void download();
			Spring *getSpring(size_t index);
			// returns 0 if Spring is uploaded to the GPU
			Spring *getDeviceSprings();
			bool getChanged() const;
			void applySpringForces(MassList &masses);
		private:
			float _ks;
			float _kd;
			std::vector<Spring> _springs;
			std::vector<unsigned int> _mass_spring_counts;
			std::vector<unsigned int> _mass_spring_indices;
			// indicates that the GPU is currently
			// computing updates for the springss
			bool _computing;
			// indicates that the spring list has
			// changed
			bool _changed;
			Spring *_device_springs;
			unsigned int *_device_mass_spring_counts;
			unsigned int *_device_mass_spring_indices;

			// CUDA
			unsigned int _threads;
	};
}
