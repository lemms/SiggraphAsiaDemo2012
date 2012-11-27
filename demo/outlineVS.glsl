/*
outlineVS.glsl
Simple pass-through shader for the screen quad.

Laurence Emms
*/

#version 400

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 uv;

out vec2 uv_v;

void main()
{
	// pass through data
	uv_v = uv;

	// vertex position
	gl_Position = position;
}

