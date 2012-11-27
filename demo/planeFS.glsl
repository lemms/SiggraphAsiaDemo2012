/*
planeFS.glsl
Fragment shader for drawing the ground plane.

Laurence Emms
*/

#version 400

in vec4 wpos_v;
in vec4 ppos_v;

layout(location = 0) out vec4 color_f;

void main()
{
	color_f = vec4(ppos_v.xyz/ppos_v.w, 1.0);
}
