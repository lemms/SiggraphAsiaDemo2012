/*
overlapFS.glsl
Fragment shader for drawing metaball overlap.

Laurence Emms
*/

#version 400

in vec4 wpos_g;
in vec4 ppos_g;
in vec2 texcoord_g;

layout(location = 0) out vec4 color_f;

uniform sampler2D depth_tex;

void main()
{
	float z_threshold = 2.0;

	vec2 screen_uv = (ppos_g.xy/ppos_g.w + vec2(1.0, 1.0))*0.5;
	vec4 depth = texture(depth_tex, screen_uv);
	
	if (abs(depth.z - ppos_g.z/ppos_g.w) > z_threshold)
		discard;

	color_f = vec4(1.0, 1.0, 1.0, 1.0);
}
