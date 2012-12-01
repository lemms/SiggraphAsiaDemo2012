/*
pointGS.glsl

Geometry shader for expanding points to quads.

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

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in float radius_v[];

out vec4 wpos_g;
out vec4 ppos_g;
out vec2 texcoord_g;

uniform mat4 Projection;

void main()
{
	vec4 top_left = vec4(radius_v[0], radius_v[0], 0.0, 0.0);
	vec4 bottom_left = vec4(radius_v[0], -radius_v[0], 0.0, 0.0);
	vec4 bottom_right = vec4(-radius_v[0], -radius_v[0], 0.0, 0.0);
	vec4 top_right = vec4(-radius_v[0], radius_v[0], 0.0, 0.0);

	wpos_g = gl_in[0].gl_Position + bottom_left;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(0.0, 0.0);
	EmitVertex();

	wpos_g = gl_in[0].gl_Position + top_left;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(0.0, 1.0);
	EmitVertex();

	wpos_g = gl_in[0].gl_Position + bottom_right;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(1.0, 0.0);
	EmitVertex();

	wpos_g = gl_in[0].gl_Position + top_right;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(1.0, 1.0);
	EmitVertex();
}
