/*
massGS.glsl
Geometry shader for expanding points to quads.

Laurence Emms
*/

#version 400

layout(points) in;
layout(quads, max_vertices = 4) out;

out vec2 texcoord_g;

uniform mat4 Projection;

void main()
{
	vec4 top_left(gl_in[0].radius_v, gl_in[0].radius_v, 0.0, 0.0);
	vec4 bottom_left(gl_in[0].radius_v, -gl_in[0].radius_v, 0.0, 0.0);
	vec4 top_right(-gl_in[0].radius_v, gl_in[0].radius_v, 0.0, 0.0);
	vec4 bottom_right(-gl_in[0].radius_v, -gl_in[0].radius_v, 0.0, 0.0);

	gl_Position =
		Projection*(gl_in[0].gl_Position + bottom_left);
	texcoord_g = vec2(0.0, 0.0);
	EmitVertex();

	gl_Position =
		Projection*(gl_in[0].gl_Position + top_left);
	texcoord_g = vec2(0.0, 1.0);
	EmitVertex();

	gl_Position =
		Projection*(gl_in[0].gl_Position + top_right);
	texcoord_g = vec2(1.0, 1.0);
	EmitVertex();

	gl_Position =
		Projection*(gl_in[0].gl_Position + bottom_right);
	texcoord_g = vec2(1.0, 0.0);
	EmitVertex();
}
