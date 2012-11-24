/*
avgFS.glsl
Fragment shader for drawing the averaging screen quad.

Laurence Emms
*/

#version 400

uniform sampler2D color_tex;
uniform sampler2D color2_tex;

uniform float inv_image_width;
uniform float inv_image_height;

in vec2 uv_v;

layout(location = 0) out vec4 color_f;

void main()
{
	//vec4 color = texture(color_tex, uv_v);
	//vec4 color2 = texture(color2_tex, uv_v);

	vec2 w = vec2(inv_image_width, 0.0) * 0.5;
	vec2 h = vec2(0.0, inv_image_height) * 0.5;
	vec3 up =		(texture(color_tex, uv_v + h).xyz +
					texture(color2_tex, uv_v + h).xyz)*0.5;
	vec3 down =		(texture(color_tex, uv_v - h).xyz +
					texture(color2_tex, uv_v - h).xyz)*0.5;
	vec3 left =		(texture(color_tex, uv_v + w).xyz +
					texture(color2_tex, uv_v + w).xyz)*0.5;
	vec3 right =	(texture(color_tex, uv_v - w).xyz +
					texture(color2_tex, uv_v - w).xyz)*0.5;

	vec3 binormal = normalize((up.xyz - down.xyz));
	vec3 tangent = normalize((left.xyz - right.xyz));
	vec3 normal = normalize(cross(tangent, binormal));

	color_f = vec4(normal, 1.0);
	//color_f = (color + color2)*0.5;
	//color_f = vec4(uv_v.xy, 0.0, 1.0);
}
