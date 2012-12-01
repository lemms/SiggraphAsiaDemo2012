/*
pointVS.glsl

Simple pass-through shader for point masses.

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

#version 400

layout(location = 0) in vec4 position;

out float radius_v;

// modelview matrix
uniform mat4 ModelView;

void main()
{
	// pass through data
	radius_v = position.w;

	// vertex position
	gl_Position = ModelView*vec4(position.xyz, 1.0);
}
