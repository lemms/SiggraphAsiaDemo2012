/*
Siggraph Asia 2012 Demo

Camera interface.

Laurence Emms
*/

namespace SigAsiaDemo {
	class Camera {
		public:
			Camera(
				float width = 1024.0,
				float height = 768.0,
				float fovy = 90.0,
				float near = 0.1,
				float far = 10000.0);
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

			glm::mat4 GetProjection();
			glm::mat4 GetModelView();
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
