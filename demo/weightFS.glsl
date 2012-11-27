/*
weightFS.glsl
Fragment shader for drawing metaball weights.

Laurence Emms
*/

#version 400

in vec4 wpos_g;
in vec4 ppos_g;
in vec2 texcoord_g;
in vec4 center_g;

layout(location = 0) out vec4 color_f;

uniform sampler2D depth_tex;
uniform sampler2D overlap_tex;

void main()
{
	float z_threshold = 10.0;

	vec2 screen_uv = (ppos_g.xy/ppos_g.w + vec2(1.0, 1.0))*0.5;
	vec4 depth = texture(depth_tex, screen_uv);
	
	if (abs(depth.z - ppos_g.z/ppos_g.w) > z_threshold) {
		color_f = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	vec3 r = wpos_g.xyz - center_g.xyz;
	float l_r = length(r);
	float l_r2 = l_r*l_r;
	float l_r3 = l_r2*l_r;

	float R = 1.0;
	float R2 = 1.0;
	float R3 = 1.0;

	float weight = 2.0*l_r3*R3 - 3.0*l_r2*R2 + 1.0;

	//float overlap = texture(overlap_tex, screen_uv).x;
	//weight /= overlap;

	//color_f = vec4(wpos_g.xyz, 1.0);
	//color_f = vec4(1.0, 0.0, 0.0, 1.0);
	color_f = vec4(weight, weight, weight, 1.0);
}
