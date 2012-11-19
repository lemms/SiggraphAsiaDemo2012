/*
screenFS.glsl
Fragment shader for drawing the screen quad.

Laurence Emms
*/

#version 400

in vec2 uv_v;

uniform sampler2D color_tex;

layout(location = 0) out vec4 color_f;

void main()
{
	//color_f = vec4(uv_v.xy, 0.0, 1.0);
	color_f = texture(color_tex, uv_v);
}
