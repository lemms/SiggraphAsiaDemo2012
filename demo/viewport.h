/*
Siggraph Asia 2012 Demo

Viewport controller interface.

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
	class Viewport {
		public:
			Viewport(
				unsigned int width = 1024,
				unsigned int height = 768,
				float near = 0.1,
				float far = 10000.0,
				float fov = 45.0);

			void ResizeWindow(
				unsigned int width,
				unsigned int height);
			void SetDistances(
				float near,
				float far);
			void SetFieldOfView(float fov);

			unsigned int GetWidth() const;
			unsigned int GetHeight() const;
			float GetAspect() const;
			float GetNear() const;
			float GetFar() const;
			float GetFieldOfView() const;
		private:
			void ComputeAspect();
			unsigned int _width;
			unsigned int _height;
			float _aspect;
			float _near;
			float _far;
			float _fov;
	};
}
