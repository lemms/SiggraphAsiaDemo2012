/*
outlineFS.glsl
Fragment shader for drawing the screen quad.

Laurence Emms
*/

#version 400

uniform sampler2D weight_tex;

in vec2 uv_v;

layout(location = 0) out vec4 color_f;

void main()
{
	vec4 weight = texture(weight_tex, uv_v);
	if (weight.y < 10.0) {
		color_f = vec4(0.0, 0.0, 0.0, 0.0);
		return;
	}
	color_f = vec4(weight.xyz, 1.0);
}
