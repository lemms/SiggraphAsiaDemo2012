/*
Siggraph Asia 2012 Demo

Mass vector implementation

Laurence Emms
*/

#include <vector>
#include "mass.h"

SigAsiaDemo::Mass::Mass(
	float x,
	float y,
	float z,
	float mass) :
		_x(x),
		_y(y),
		_z(z),
		_mass(mass)
{}

SigAsiaDemo::MassList::MassList() {}
