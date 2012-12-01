/*
Siggraph Asia 2012 Demo

CUDA device implementation.

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

#include <iostream>

// GLM
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "camera.h"

SigAsiaDemo::Camera::Camera(
	float width,
	float height,
	float fovy,
	float near,
	float far) :
		_fovy(fovy),
		_aspect(width/height),
		_near(near),
		_far(far)
{
	_l.x = 0.0; _l.y = 0.0; _l.z = 0.0;
	_p.x = 10.0; _p.y = 20.0; _p.z = 50.0;
	_u.x = 0.0; _u.y = 1.0; _u.z = 0.0;
}

void SigAsiaDemo::Camera::SetLook(
	float x, float y, float z)
{
	_l.x = x;
	_l.y = y;
	_l.z = z;
}

void SigAsiaDemo::Camera::MoveLook(
	float x, float y, float z)
{
	_l.x += x;
	_l.y += y;
	_l.z += z;
}

void SigAsiaDemo::Camera::SetPosition(
	float x, float y, float z)
{
	_p.x = x;
	_p.y = y;
	_p.z = z;
}

void SigAsiaDemo::Camera::MovePosition(
	float x, float y, float z)
{
	_p.x += x;
	_p.y += y;
	_p.z += z;
}

void SigAsiaDemo::Camera::SetUp(
	float x, float y, float z)
{
	_u.x = x;
	_u.y = y;
	_u.z = z;
}

void SigAsiaDemo::Camera::ResizeWindow(
	float width, float height)
{
	if (height == 0.0)
		height = 1.0;
	_aspect = width/height;
}

glm::vec3 SigAsiaDemo::Camera::GetLook() const
{
	return _l;
}

glm::vec3 SigAsiaDemo::Camera::GetPosition() const
{
	return _p;
}

glm::vec3 SigAsiaDemo::Camera::GetUp() const
{
	return _u;
}

glm::mat4 SigAsiaDemo::Camera::GetProjection()
{
	return glm::perspective(
		_fovy,
		_aspect,
		_near,
		_far);
}

glm::mat4 SigAsiaDemo::Camera::GetModelView()
{
	return glm::lookAt(_p, _l, _u);
}

glm::mat3 SigAsiaDemo::Camera::GetNormal()
{
	return glm::inverseTranspose(glm::mat3(glm::lookAt(_p, _l, _u)));
}
