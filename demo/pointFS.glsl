/*
pointFS.glsl

Fragment shader for drawing quads as points.

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

in vec4 wpos_g;
in vec4 ppos_g;
in vec2 texcoord_g;

layout(location = 0) out vec4 color_f;

void main()
{
	float r = length(texcoord_g - vec2(0.5, 0.5));
	// discard beyond radius 0.2 to create points
	if (r > 0.1)
		discard;
	
	color_f = vec4(1.0, 0.0, 0.0, 1.0);
}
