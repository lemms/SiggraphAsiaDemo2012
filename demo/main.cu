/*
Siggraph Asia 2012 Demo

Laurence Emms
*/

#include <iostream>
#include <string>
#include <sstream>
using namespace std;

// OpenGL
#ifdef WIN32
#include <windows.h>
#define GLEW_STATIC
#define GLUT_DISABLE_ATEXIT_HACK
#endif
#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif

// GLM
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

// demo
#include "device.h"
#include "viewport.h"
#include "mass.h"
#include "spring.h"
#include "camera.h"
#include "creator.h"
#include "cube.h"

// globals
SigAsiaDemo::Viewport viewport;
SigAsiaDemo::Camera camera;
SigAsiaDemo::MassList masses;
SigAsiaDemo::SpringList springs;
SigAsiaDemo::CubeList cubes;

size_t frame = 0;
float dt = 1e-5;
bool play = false;
bool ground_collision = true;

float resolution_multiplier = 0.02;
float cube_spacing = 1.0;
float cube_radius = 1.0;

void ParseArgs(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i) {
		unsigned int width = viewport.GetWidth();
		unsigned int height = viewport.GetHeight();
		stringstream stream(argv[i]);
		if (stream.str() == "-") {
			cout << "usage: SiggraphAsiaDemo \
[-w width] [-h height]" << endl;
			exit(0);
		}
		if (stream.str() == "-w") {
			if (i++ > argc)
				return;
			stream.str(argv[i]);
			cout << stream.str() << endl;
			stream >> width;
		}
		if (stream.str() == "-h") {
			if (i++ > argc)
				return;
			stream.str(argv[i]);
			cout << stream.str() << endl;
			stream >> height;
		}
		viewport.ResizeWindow(width, height);
		camera.ResizeWindow(width, height);
		glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());
	}
}

void PrintMassesAndSprings()
{
	masses.download();

	for (size_t i = 0; i < masses.size(); i++) {
		SigAsiaDemo::Mass *mass = masses.getMass(i);
		if (mass) {
			cout << "Point Mass " << i << endl;
			cout << "mass: " << mass->_mass << endl;
			cout << "position: (";
			cout << mass->_x << ", ";
			cout << mass->_y << ", ";
			cout << mass->_z << ")" << endl;
			cout << "velocity: (";
			cout << mass->_vx << ", ";
			cout << mass->_vy << ", ";
			cout << mass->_vz << ")" << endl;
			cout << "temporary position: (";
			cout << mass->_tx << ", ";
			cout << mass->_ty << ", ";
			cout << mass->_tz << ")" << endl;
			cout << "temporary velocity: (";
			cout << mass->_tvx << ", ";
			cout << mass->_tvy << ", ";
			cout << mass->_tvz << ")" << endl;
			cout << "velocity k1: (";
			cout << mass->_k1x << ", ";
			cout << mass->_k1y << ", ";
			cout << mass->_k1z << ")" << endl;
			cout << "velocity k2: (";
			cout << mass->_k2x << ", ";
			cout << mass->_k2y << ", ";
			cout << mass->_k2z << ")" << endl;
			cout << "velocity k3: (";
			cout << mass->_k3x << ", ";
			cout << mass->_k3y << ", ";
			cout << mass->_k3z << ")" << endl;
			cout << "velocity k4: (";
			cout << mass->_k4x << ", ";
			cout << mass->_k4y << ", ";
			cout << mass->_k4z << ")" << endl;
			cout << "force: (";
			cout << mass->_fx << ", ";
			cout << mass->_fy << ", ";
			cout << mass->_fz << ")" << endl;
			cout << "radius: ";
			cout << mass->_radius << endl;
			cout <<endl;
		}
	}

	springs.download();

	for (size_t i = 0; i < springs.size(); i++) {
		SigAsiaDemo::Spring *spring = springs.getSpring(i);
		if (spring) {
			cout << "Spring " << i << endl;
			cout << "mass 0: " << spring->_mass0 << endl;
			cout << "mass 1: " << spring->_mass1 << endl;
			cout << "l0: " << spring->_l0 << endl;
			cout << "f0: (";
			cout << spring->_fx0 << ", ";
			cout << spring->_fy0 << ", ";
			cout << spring->_fz0 << ")" << endl;
			cout << "f1: (";
			cout << spring->_fx1 << ", ";
			cout << spring->_fy1 << ", ";
			cout << spring->_fz1 << ")" << endl;
			cout <<endl;
		}
	}
}

void Step()
{
	cubes.computeBounds(masses);
	cubes.collideCubes(dt, masses);

	masses.upload();
	springs.upload(masses, cubes);

	masses.startFrame();

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK1(dt, ground_collision);

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK2(dt, ground_collision);

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK3(dt, ground_collision);

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK4(dt, ground_collision);

	masses.update(dt, ground_collision);
	masses.endFrame();

	//PrintMassesAndSprings();

	frame++;
}

void Idle()
{
	if (play) {
		Step();
		glutPostRedisplay();
	}
}

void Reshape(int width, int height)
{
	viewport.ResizeWindow(width, height);
	camera.ResizeWindow(width, height);
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());
}

bool render_points = false;
void Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (render_points)
		masses.render(camera.GetModelView(), camera.GetProjection());
	else
		cubes.render(camera.GetModelView(), camera.GetProjection(), camera.GetNormal());

	glutSwapBuffers();
}

void Keys(unsigned char key, int x, int y)
{
	float px = 0.0, py = 0.0, pz = 0.0;
	if (key == 'p') render_points = !render_points;
	if (key == 'q') px = 10.0;
	if (key == 'a') px = -10.0;
	if (key == 'w') py = 10.0;
	if (key == 's') py = -10.0;
	if (key == 'e') pz = 10.0;
	if (key == 'd') pz = -10.0;
	if (px != 0.0 || py != 0.0 || pz != 0.0) {
		unsigned int width = viewport.GetWidth();
		unsigned int height = viewport.GetHeight();
		glutPostRedisplay();
	}
	camera.MovePosition(px, py, pz);

	if (key == 'c')
		play = true;
	if (key == 'v')
		play = false;

	if (key == 'z') {
		Step();
		glutPostRedisplay();
	}

	if (key == 27) {
		exit(0);
	}
}
void SpecialKeys(int key, int x, int y)
{
	if (key == GLUT_KEY_F1) {
		cout << "Controls: " << endl;
		cout << "Move camera x: q/a" << endl;
		cout << "Move camera y: w/s" << endl;
		cout << "Move camera z: e/d" << endl;
		cout << "Play simulation: c" << endl;
		cout << "Stop simulation: v" << endl;
		cout << "Step simulation: z" << endl;
		cout << "Esc to exit" << endl;
	}
}

int main(int argc, char **argv)
{
	ParseArgs(argc, argv);
	cout << "Siggraph Asia 2012 Demo" << endl;
	cout << "Author: Laurence Emms" << endl;

	int device = 0;
	{
		int result = SigAsiaDemo::setDevice(device);
		if (result != 0)
			return result;
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
		cerr << "Error: Failed to initialize GLEW." << endl;
		return 1;
	}
	cout << "Using GLEW " << glewGetString(GLEW_VERSION) \
	<< "." << endl;
	if (!GLEW_VERSION_4_2)
	{
		cerr << "Error: OpenGL 4.2 not supported." << endl;
		return 1;
	}
	cout << "Using OpenGL " << glGetString(GL_VERSION) << "." \
	<< endl;


	// set CUDA/OpenGL device
	cout << "Set CUDA/OpenGL device." << endl;
	SigAsiaDemo::setGLDevice(device);

	// load shaders
	cout << "Load shaders." << endl;
	masses.loadShaders();
	cubes.loadShaders();

	// initialize OpenGL
	cout << "Initialize OpenGL." << endl;
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.7, 0.7, 1.0, 1.0);

	// resize viewport
	cout << "Resize viewport." << endl;
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());

	// fill masses
	cout << "Fill masses." << endl;

	springs.setConstants(10000.0, 1000.0);
	cubes.setConstants(10000.0, 1000.0);

	cout << "Setup cube." << endl;

	cubes.push(SigAsiaDemo::Cube(
		-40.0, 20.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		-20.0, 30.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		0.0, 40.0, 0.0,		// position
		10, 10, 10,			// size
		cube_spacing*0.5,	// spacing
		4.0,				// mass
		cube_radius*0.5	// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		20.0, 50.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		40.0, 60.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));

	cubes.push(SigAsiaDemo::Cube(
		-40.0, 40.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		-20.0, 50.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		0.0, 60.0, 0.0,		// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		20.0, 70.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	cubes.push(SigAsiaDemo::Cube(
		40.0, 80.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));

	cout << "Create cubes." << endl;
	cubes.create(masses, springs);

	cout << "Added " << masses.size() << " masses." << endl;
	cout << "Added " << springs.size() << " springs." << endl;
	cout << "Creation complete." << endl;

	cout << "Initialize masses." << endl;
	Step();
	cout << "Initialization complete." << endl;

	// register callbacks
	cout << "Register callbacks." << endl;
	glutIdleFunc(Idle);
	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Keys);
	glutSpecialFunc(SpecialKeys);

	// enter GLUT event processing cycle
	cout << "Enter GLUT event processing cycle." << endl;
	cout << endl;
	cout << "=================" << endl;
	cout << "Press F1 for help" << endl;
	cout << "=================" << endl;
	cout << endl;
	glutMainLoop();

	return 0;
}
