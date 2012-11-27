/*
depthFS.glsl
Fragment shader for drawing quad depths.

Laurence Emms
*/

#version 400

in vec4 wpos_g;
in vec4 ppos_g;
in vec2 texcoord_g;

layout(location = 0) out vec4 color_f;

void main()
{
	color_f = vec4(ppos_g.xyz/ppos_g.w, 1.0);
}
