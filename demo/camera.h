/*
Siggraph Asia 2012 Demo

Camera interface.

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
	class Camera {
		public:
			Camera(
				float width = 1024.0,
				float height = 768.0,
				float fovy = 90.0,
				float near = 0.1,
				float far = 1000.0);
			void SetLook(
				float x, float y, float z);
			void MoveLook(
				float x, float y, float z);
			void SetPosition(
				float x, float y, float z);
			void MovePosition(
				float x, float y, float z);
			void SetUp(
				float x, float y, float z);
			void ResizeWindow(
				float width, float height);

			glm::vec3 GetLook() const;
			glm::vec3 GetPosition() const;
			glm::vec3 GetUp() const;

			glm::mat4 GetProjection();
			glm::mat4 GetModelView();
			glm::mat3 GetNormal();
		private:
			// projection
			float _fovy;
			float _aspect;
			float _near;
			float _far;
			
			// look vector
			glm::vec3 _l;
			// position vector
			glm::vec3 _p;
			// up vector
			glm::vec3 _u;
	};
}
