/*
Siggraph Asia 2012 Demo

Spring vector implementation

Laurence Emms
*/

#include <vector>
#include "mass.h"
#include "spring.h"

SigAsiaDemo::Spring::Spring(
	unsigned int mass0,
	unsigned int mass1,
	float ks,
	float kd) :
		_mass0(mass0),
		_mass1(mass1),
		_ks(ks),
		_kd(kd),
		_l0(0.0)
{}

SigAsiaDemo::SpringList::SpringList()
{
}
