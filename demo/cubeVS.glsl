/*
cubeVS.glsl
Simple pass-through shader for cube surface.

Laurence Emms
*/

#version 400

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

// modelview matrix
uniform mat4 ModelView;
uniform mat4 Projection;
uniform mat3 Normal;

out vec4 wpos_v;
out vec3 wnorm_v;

void main()
{
	wpos_v = ModelView*vec4(position, 1.0);
	wnorm_v = Normal*normal;
	
	// vertex position
	gl_Position = Projection*wpos_v;
}

