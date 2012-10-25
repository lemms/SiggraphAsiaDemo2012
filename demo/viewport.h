/*
Siggraph Asia 2012 Demo

Viewport controller

Laurence Emms
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
