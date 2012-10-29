/*
massVS.glsl
Simple pass-through shader for point masses.

Laurence Emms
*/

#version 400

layout(location = 0) in vec4 position;

out float radius_v;

// modelview matrix
uniform mat4 ModelView;
// projection matrix
uniform mat4 Projection;

void main()
{
	// pass through data
	// TODO
	//radius_v = position.w;
	radius_v = 1.0;

	// vertex position
	gl_Position = ModelView*vec4(position.xyz, 1.0);
}

