/*
cubeFS.glsl
Fragment shader for drawing cube surface.

Laurence Emms
*/

#version 400

in vec4 wpos_v;
in vec3 wnorm_v;

layout(location = 0) out vec4 color_f;

void main()
{
	color_f = vec4(wnorm_v.xyz, 1.0);
}
