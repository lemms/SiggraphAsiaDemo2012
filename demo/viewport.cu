/*
Siggraph Asia 2012 Demo

Viewport controller implementation.

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

void SigAsiaDemo::Viewport::ResizeWindow(
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
