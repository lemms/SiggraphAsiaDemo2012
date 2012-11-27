/*
planeVS.glsl
Simple pass-through shader for the ground plane.

Laurence Emms
*/

#version 400

layout(location = 0) in vec4 position;

// modelview matrix
uniform mat4 ModelView;
uniform mat4 Projection;

out vec4 wpos_v;
out vec4 ppos_v;

void main()
{
	// pass through data
	wpos_v = ModelView*vec4(position.xyz, 1.0);

	// vertex position
	ppos_v = Projection*wpos_v;
	gl_Position = ppos_v;
}

