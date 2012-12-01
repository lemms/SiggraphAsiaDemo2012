/*
cubeVS.glsl

Simple pass-through shader for cube surface.

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

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

// modelview matrix
uniform mat4 ModelView;
uniform mat4 Projection;
uniform mat3 Normal;

out vec4 wpos_v;
out vec3 norm_v;
out vec3 color_v;

void main()
{
	wpos_v = ModelView*vec4(position, 1.0);
	norm_v = normal;
	color_v = color;
	
	// vertex position
	gl_Position = Projection*wpos_v;
}

