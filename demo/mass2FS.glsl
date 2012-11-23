/*
mass2FS.glsl
Fragment shader for drawing gaussian particles.
Discards particles at the depth of the input image.

Laurence Emms
*/

#version 400

uniform float inv_width;
uniform float inv_height;

uniform sampler2D color_tex;

in vec4 wpos_g;
in vec4 ppos_g;
in vec2 texcoord_g;

layout(location = 0) out vec4 color_f;

void main()
{
	vec2 r = texcoord_g - vec2(0.5, 0.5);
	float l_r = length(r);
	if (l_r > 0.5f) {
		discard;
	}

	vec2 screen_uv = (ppos_g.xy/ppos_g.w + vec2(1.0, 1.0))*0.5;
	vec4 tex_col = texture(color_tex, screen_uv);

	float dist = abs(wpos_g.z - tex_col.z)/100.0;
	float epsilon = 0.02;
	if (dist > epsilon) {
		discard;
	}

	//color_f = vec4(dist, dist, dist, 1.0);
	color_f = vec4(l_r, l_r, l_r, 1.0);
	//color_f = vec4(wpos_g.xyz, 1.0);
}
