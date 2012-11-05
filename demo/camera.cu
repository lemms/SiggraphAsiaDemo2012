/*
Siggraph Asia 2012 Demo

CUDA device implementation.

Laurence Emms
*/

#include <iostream>

// GLM
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
	_p.x = 0.0; _p.y = 0.0; _p.z = 30.0;
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
