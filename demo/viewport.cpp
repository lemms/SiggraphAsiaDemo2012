#include "viewport.h"

SigAsiaDemo::Viewport::Viewport(
	unsigned int width,
	unsigned int height,
	float near,
	float far,
	float fov) :
	_width(width),
	_height(height),
	_aspect(0.0),
	_near(near),
	_far(far),
	_fov(fov)
{
	ComputeAspect();
} 

void SigAsiaDemo::Viewport::SetDimensions(
	unsigned int width,
	unsigned int height)
{
	_width = (width > 0) ? width : 1;
	_height = (height > 0) ? height : 1;
	ComputeAspect();
}
void SigAsiaDemo::Viewport::SetDistances(
	float near,
	float far)
{
	_near = near;
	_far = far;
}
void SigAsiaDemo::Viewport::SetFieldOfView(
	float fov)
{
	_fov = fov;
}

unsigned int SigAsiaDemo::Viewport::GetWidth() const
{
	return _width;
}
unsigned int SigAsiaDemo::Viewport::GetHeight() const
{
	return _height;
}
float SigAsiaDemo::Viewport::GetAspect() const
{
	return _aspect;
}
float SigAsiaDemo::Viewport::GetNear() const
{
	return _near;
}
float SigAsiaDemo::Viewport::GetFar() const
{
	return _far;
}
float SigAsiaDemo::Viewport::GetFieldOfView() const
{
	return _fov;
}

void SigAsiaDemo::Viewport::ComputeAspect()
{
	_aspect = static_cast<float>(_width)/static_cast<float>(_height);
}
