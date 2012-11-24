/*
screenFS.glsl
Fragment shader for drawing the screen quad.

Laurence Emms
*/

#version 400

uniform sampler2D position_tex;
uniform sampler2D normal_tex;

uniform float inv_image_width;
uniform float inv_image_height;

in vec2 uv_v;

layout(location = 0) out vec4 color_f;

void main()
{
	vec4 p = texture(position_tex, uv_v);
	if (p.w == -10.0)
		discard;

	// define a single directional light
	vec3 ldir = vec3(0.5, 0.5, 0.5);

	float w = inv_image_width;
	float h = inv_image_height;

	vec3 position = 
		texture(position_tex, uv_v +
			vec2(w, h)).xyz * 0.0625 +
		texture(position_tex, uv_v +
			vec2(w, -h)).xyz * 0.0625 +
		texture(position_tex, uv_v +
			vec2(-w, h)).xyz * 0.0625 +
		texture(position_tex, uv_v +
			vec2(-w, -h)).xyz * 0.0625 +
		texture(position_tex, uv_v +
			vec2(w, 0.0)).xyz * 0.125 +
		texture(position_tex, uv_v +
			vec2(-w, 0.0)).xyz * 0.125 +
		texture(position_tex, uv_v +
			vec2(0.0, h)).xyz * 0.125 +
		texture(position_tex, uv_v +
			vec2(0.0, -h)).xyz * 0.125 +
		p.xyz * 0.25;

	vec3 normal =
		texture(normal_tex, uv_v +
			vec2(w, h)).xyz * 0.0625 +
		texture(normal_tex, uv_v +
			vec2(w, -h)).xyz * 0.0625 +
		texture(normal_tex, uv_v +
			vec2(-w, h)).xyz * 0.0625 +
		texture(normal_tex, uv_v +
			vec2(-w, -h)).xyz * 0.0625 +
		texture(normal_tex, uv_v +
			vec2(w, 0.0)).xyz * 0.125 +
		texture(normal_tex, uv_v +
			vec2(-w, 0.0)).xyz * 0.125 +
		texture(normal_tex, uv_v +
			vec2(0.0, h)).xyz * 0.125 +
		texture(normal_tex, uv_v +
			vec2(0.0, -h)).xyz * 0.125 +
		texture(normal_tex, uv_v).xyz * 0.25;

	float diffuse = dot(ldir, normal);

	//color_f = vec4(uv_v.xy, 0.0, 1.0);
	//color_f = vec4(normal, 1.0);
	//color_f = vec4(position, 1.0);
	color_f = vec4(diffuse, diffuse, diffuse, 1.0);
}
