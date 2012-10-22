/*
massFS.glsl
Fragment shader for drawing gaussian particles.

Laurence Emms
*/

#version 400

in vec3 texcoord_g;

layout(location = 0) out vec4 color_f;

void main()
{
	// TODO: add gaussian

	// Temporarily draw UVs
	color_f = vec4(texcoord_g.x, texcoord_g.y, 0.0, 1.0);
}
