/*
Siggraph Asia 2012 Demo

Laurence Emms
*/

#include <iostream>
#include <string>
#include <sstream>
using namespace std;

// OpenGL
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif

// TODO: remove and swap to programmable pipeline
// temporarily including glu
#include <GL/glu.h>

// CUDA
#include <cuda.h>

// demo
#include "viewport.h"
#include "mass.h"

// globals
SigAsiaDemo::Viewport viewport;
SigAsiaDemo::MassList masses;

size_t frame = 0;

void ParseArgs(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i) {
		unsigned int width = viewport.GetWidth();
		unsigned int height = viewport.GetHeight();
		stringstream stream(argv[i]);
		if (stream.str() == "-") {
			std::cout << "usage: SiggraphAsiaDemo \
[-w width] [-h height]" << std::endl;
			exit(0);
		}
		if (stream.str() == "-w") {
			if (i++ > argc)
				return;
			stream.str(argv[i]);
			std::cout << stream.str() << std::endl;
			stream >> width;
		}
		if (stream.str() == "-h") {
			if (i++ > argc)
				return;
			stream.str(argv[i]);
			std::cout << stream.str() << std::endl;
			stream >> height;
		}
		viewport.SetDimensions(width, height);
	}
}

void Idle()
{
	std::cout << "Frame: " << frame << std::endl;
	masses.upload();
	masses.update(0.1);
	//masses.download();

	frame++;
}

void Reshape(int width, int height)
{
	viewport.SetDimensions(width, height);
	// TODO: remove and swap to programmable pipeline
	// resize viewport
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(
		viewport.GetFieldOfView(),
		viewport.GetAspect(),
		viewport.GetNear(),
		viewport.GetFar());

	// Get back to the Modelview
	glMatrixMode(GL_MODELVIEW);
}

void Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBegin(GL_TRIANGLES);
	glVertex3f(-0.5,-0.5,0.0);
	glVertex3f(0.5,0.0,0.0);
	glVertex3f(0.0,0.5,0.0);
	glEnd();

	glutSwapBuffers();
}

void Keys(unsigned char key, int x, int y)
{
	if (key == 27 || key == 'q') {
		// TODO: remove this printout
		masses.download();
		for (size_t i = 0; i < masses.size(); i++) {
			SigAsiaDemo::Mass *mass0 = masses.getMass(i);
			if (mass0) {
				std::cout << "Point Mass " << i << std::endl;
				std::cout << "mass: " << mass0->_mass << std::endl;
				std::cout << "position: (";
				std::cout << mass0->_x << ", ";
				std::cout << mass0->_y << ", ";
				std::cout << mass0->_z << ")" << std::endl;
				std::cout << "velocity: (";
				std::cout << mass0->_vx << ", ";
				std::cout << mass0->_vy << ", ";
				std::cout << mass0->_vz << ")" << std::endl;
				std::cout << "acceleration: (";
				std::cout << mass0->_ax << ", ";
				std::cout << mass0->_ay << ", ";
				std::cout << mass0->_az << ")" << std::endl;
				std::cout <<std::endl;
			}
		}
		exit(0);
	}
}
void SpecialKeys(int key, int x, int y)
{
	if (key == GLUT_KEY_F1)
		// TODO: display help text
		std::cout << "Help here" << std::endl;
}

int main(int argc, char **argv)
{
	ParseArgs(argc, argv);
	cout << "Siggraph Asia 2012 Demo" << endl;
	cout << "Author: Laurence Emms" << endl;

	// get cuda device properties
	cudaDeviceProp properties;
	int count;
	cudaGetDeviceCount(&count);
	if (count <= 0) {
		std::cerr << "Error: No CUDA devices found." << std::endl;
		return 1;
	}
	for (int i = 0; i < count; ++i) {
		std::cout << "CUDA Device " << i << ":" << std::endl;
		cudaGetDeviceProperties(&properties, i);
		std::cout << "Name: " << properties.name << std::endl;
		std::cout << "Compute Capability: " \
		<< properties.major << "." << properties.minor << std::endl;
		std::cout << "Global Memory: " \
		<< properties.totalGlobalMem << std::endl;
		std::cout << "Constant Memory: " \
		<< properties.totalConstMem << std::endl;
		std::cout << "Shared Memory: " \
		<< properties.sharedMemPerBlock << std::endl;
		std::cout << "Registers: " \
		<< properties.regsPerBlock << std::endl;
		std::cout << "Warp Size: " \
		<< properties.warpSize << std::endl;
		std::cout << "Max Threads Per Block: " \
		<< properties.maxThreadsPerBlock << std::endl;
		std::cout << "Max Threads: (" \
		<< properties.maxThreadsDim[0] << ", " \
		<< properties.maxThreadsDim[1] << ", " \
		<< properties.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Max Grid Size: (" \
		<< properties.maxGridSize[0] << ", " \
		<< properties.maxGridSize[1] << ", " \
		<< properties.maxGridSize[2] << ")" << std::endl;
	}

	// setup GLUT
	glutInit(&argc, argv);
#ifdef FREEGLUT
	cout << "Using FreeGLUT" << endl;
	if (glutGet(GLUT_VERSION) < 20001) {
		cout << "Sorry, you need freeglut version 2.0.1 or later to \
			run this program." << endl;
		return 1;
	}
#else
	cout << "Sorry, you need freeglut version 2.0.1 or later to \
		run this program." << endl;
#endif
	glutInitDisplayMode(
			GLUT_DEPTH | 
			GLUT_DOUBLE | 
			GLUT_RGBA | 
			GLUT_MULTISAMPLE);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(viewport.GetWidth(), viewport.GetHeight());
	glutCreateWindow("Siggraph Asia 2012 Mass Spring Demo");

	// initialize GLEW
	GLenum result = glewInit();
	if (result != GLEW_OK) {
		std::cerr << "Error: Failed to initialize GLEW." << std::endl;
		return 1;
	}
	std::cout << "Using GLEW " << glewGetString(GLEW_VERSION) << "." << std::endl;
	if (!GLEW_VERSION_4_2)
	{
		std::cerr << "Error: OpenGL 4.2 not supported." << std::endl;
		return 1;
	}
	std::cout << "Using OpenGL " << glGetString(GL_VERSION) << "." << std::endl;

	// initialize OpenGL
	glClearColor(0.0, 0.0, 0.0, 0.0);

	// resize viewport
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());

	// setup projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(
		viewport.GetFieldOfView(),
		viewport.GetAspect(),
		viewport.GetNear(),
		viewport.GetFar());

	// Get back to the Modelview
	glMatrixMode(GL_MODELVIEW);

	// fill masses
	for (int i = 0; i < 1000000; i++)
		masses.push(SigAsiaDemo::Mass(1.0));

	// register callbacks
	glutIdleFunc(Idle);
	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Keys);
	glutSpecialFunc(SpecialKeys);

	// enter GLUT event processing cycle
	glutMainLoop();

	return 0;
}
