/*
pointFS.glsl
Fragment shader for drawing quads as points.

Laurence Emms
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
