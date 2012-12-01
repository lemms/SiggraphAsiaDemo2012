/*
cubeFS.glsl

Fragment shader for drawing cube surface.

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

in vec4 wpos_v;
in vec3 norm_v;
in vec3 color_v;

uniform mat4 ModelView;
uniform mat3 Normal;

layout(location = 0) out vec4 color_f;

void main()
{
	vec3 light_dir = normalize((ModelView*vec4(-1.0, -1.0, -1.0, 1.0)).xyz);
	vec3 n = normalize(Normal*norm_v);

	float ndotl = max(dot(n, light_dir), 0.0);
	vec3 a = vec3(0.5, 0.5, 0.5);
	vec3 d = vec3(ndotl, ndotl, ndotl);

	//color_f = vec4(wnorm_v.xyz, 1.0);
	color_f = vec4(color_v*(a+d), 1.0);
}
