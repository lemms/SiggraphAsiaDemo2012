/*
massGS.glsl
Geometry shader for expanding points to quads.

Laurence Emms
*/

#version 400

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in float radius_v[];

out vec4 wpos_g;
out vec4 ppos_g;
out vec2 texcoord_g;

uniform mat4 Projection;

void main()
{
	vec4 top_left = vec4(radius_v[0], radius_v[0], 0.0, 0.0);
	vec4 bottom_left = vec4(radius_v[0], -radius_v[0], 0.0, 0.0);
	vec4 bottom_right = vec4(-radius_v[0], -radius_v[0], 0.0, 0.0);
	vec4 top_right = vec4(-radius_v[0], radius_v[0], 0.0, 0.0);

	wpos_g = gl_in[0].gl_Position + bottom_left;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(0.0, 0.0);
	EmitVertex();

	wpos_g = gl_in[0].gl_Position + top_left;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(0.0, 1.0);
	EmitVertex();

	wpos_g = gl_in[0].gl_Position + bottom_right;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(1.0, 0.0);
	EmitVertex();

	wpos_g = gl_in[0].gl_Position + top_right;
	ppos_g = Projection*wpos_g;
	gl_Position = ppos_g;
	texcoord_g = vec2(1.0, 1.0);
	EmitVertex();
}
