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
#pragma (lib, "freeglut_static.lib")
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
		viewport.ResizeWindow(width, height);
		camera.ResizeWindow(width, height);
		masses.resizeWindow(
			viewport.GetNear(),
			viewport.GetFar(),
			viewport.GetFieldOfView(),
			glm::length(camera.GetLook()-camera.GetPosition()),
			cube_spacing*resolution_multiplier,
			width, height);
		glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());
	}
}

void PrintMassesAndSprings()
{
	masses.download();

	for (size_t i = 0; i < masses.size(); i++) {
		SigAsiaDemo::Mass *mass = masses.getMass(i);
		if (mass) {
			std::cout << "Point Mass " << i << std::endl;
			std::cout << "mass: " << mass->_mass << std::endl;
			std::cout << "position: (";
			std::cout << mass->_x << ", ";
			std::cout << mass->_y << ", ";
			std::cout << mass->_z << ")" << std::endl;
			std::cout << "velocity: (";
			std::cout << mass->_vx << ", ";
			std::cout << mass->_vy << ", ";
			std::cout << mass->_vz << ")" << std::endl;
			std::cout << "temporary position: (";
			std::cout << mass->_tx << ", ";
			std::cout << mass->_ty << ", ";
			std::cout << mass->_tz << ")" << std::endl;
			std::cout << "temporary velocity: (";
			std::cout << mass->_tvx << ", ";
			std::cout << mass->_tvy << ", ";
			std::cout << mass->_tvz << ")" << std::endl;
			std::cout << "velocity k1: (";
			std::cout << mass->_k1x << ", ";
			std::cout << mass->_k1y << ", ";
			std::cout << mass->_k1z << ")" << std::endl;
			std::cout << "velocity k2: (";
			std::cout << mass->_k2x << ", ";
			std::cout << mass->_k2y << ", ";
			std::cout << mass->_k2z << ")" << std::endl;
			std::cout << "velocity k3: (";
			std::cout << mass->_k3x << ", ";
			std::cout << mass->_k3y << ", ";
			std::cout << mass->_k3z << ")" << std::endl;
			std::cout << "velocity k4: (";
			std::cout << mass->_k4x << ", ";
			std::cout << mass->_k4y << ", ";
			std::cout << mass->_k4z << ")" << std::endl;
			std::cout << "force: (";
			std::cout << mass->_fx << ", ";
			std::cout << mass->_fy << ", ";
			std::cout << mass->_fz << ")" << std::endl;
			std::cout << "radius: ";
			std::cout << mass->_radius << std::endl;
			std::cout <<std::endl;
		}
	}

	springs.download();

	for (size_t i = 0; i < springs.size(); i++) {
		SigAsiaDemo::Spring *spring = springs.getSpring(i);
		if (spring) {
			std::cout << "Spring " << i << std::endl;
			std::cout << "mass 0: " << spring->_mass0 << std::endl;
			std::cout << "mass 1: " << spring->_mass1 << std::endl;
			std::cout << "l0: " << spring->_l0 << std::endl;
			std::cout << "f0: (";
			std::cout << spring->_fx0 << ", ";
			std::cout << spring->_fy0 << ", ";
			std::cout << spring->_fz0 << ")" << std::endl;
			std::cout << "f1: (";
			std::cout << spring->_fx1 << ", ";
			std::cout << spring->_fy1 << ", ";
			std::cout << spring->_fz1 << ")" << std::endl;
			std::cout <<std::endl;
		}
	}
}

void Step()
{
	//std::cout << "Frame: " << frame << std::endl;
	//std::cout << "compute bounds." << std::endl;
	cubes.computeBounds(masses);
	cubes.collideCubes(dt, masses);
	/*
	// print bounds
	for (size_t i = 0; i < cubes.size(); ++i) {
		SigAsiaDemo::Cube *cube = cubes.getCube(i);
		if (cube) {
			std::cout << "Cube " << i << ":" << std::endl;
			std::cout << "[" << cube->_min_x << ", " \
			<< cube->_max_x << "]" << std::endl;
			std::cout << "[" << cube->_min_y << ", " \
			<< cube->_max_y << "]" << std::endl;
			std::cout << "[" << cube->_min_z << ", " \
			<< cube->_max_z << "]" << std::endl;
		}
	}
	*/

	//std::cout << "upload masses." << std::endl;
	masses.upload();
	//std::cout << "upload springs." << std::endl;
	springs.upload(masses);

	//std::cout << "start frame." << std::endl;
	masses.startFrame();

	//std::cout << "evaluate k1." << std::endl;
	//std::cout << "clear forces." << std::endl;
	masses.clearForces();
	//std::cout << "apply spring forces." << std::endl;
	springs.applySpringForces(masses);
	//std::cout << "evaluate." << std::endl;
	masses.evaluateK1(dt, ground_collision);

	//std::cout << "evaluate k2." << std::endl;
	//std::cout << "clear forces." << std::endl;
	masses.clearForces();
	//std::cout << "apply spring forces." << std::endl;
	springs.applySpringForces(masses);
	//std::cout << "evaluate." << std::endl;
	masses.evaluateK2(dt, ground_collision);

	//std::cout << "evaluate k3." << std::endl;
	//std::cout << "clear forces." << std::endl;
	masses.clearForces();
	//std::cout << "apply spring forces." << std::endl;
	springs.applySpringForces(masses);
	//std::cout << "evaluate." << std::endl;
	masses.evaluateK3(dt, ground_collision);

	//std::cout << "evaluate k4." << std::endl;
	//std::cout << "clear forces." << std::endl;
	masses.clearForces();
	//std::cout << "apply spring forces." << std::endl;
	springs.applySpringForces(masses);
	//std::cout << "evaluate." << std::endl;
	masses.evaluateK4(dt, ground_collision);

	//std::cout << "update." << std::endl;
	masses.update(dt, ground_collision);
	masses.endFrame();

	//PrintMassesAndSprings();

	frame++;
}

void Idle()
{
	if (play) {
		Step();
		//std::cout << "Stepping, frame: " << frame << std::endl;
		// TODO: come up with a better metric
		if (frame % 5 == 0)
			glutPostRedisplay();
	}

	// TODO: remove
	/*
	if (frame == 1000) {
		PrintMassesAndSprings();

		exit(0);
	}
	*/
}

void Reshape(int width, int height)
{
	viewport.ResizeWindow(width, height);
	camera.ResizeWindow(width, height);
	masses.resizeWindow(
		viewport.GetNear(),
		viewport.GetFar(),
		viewport.GetFieldOfView(),
		glm::length(camera.GetLook()-camera.GetPosition()),
		cube_spacing*resolution_multiplier,
		width, height);
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());
}

void Render()
{
	//std::cout << "Render" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	/*
	const float *MV = glm::value_ptr(camera.GetModelView());
	const float *P = glm::value_ptr(camera.GetProjection());
	std::cout << "MV:" << std::endl;
	std::cout << "[" << MV[0]  << " " << MV[1]  <<  " " << MV[2]  <<  " " << MV[3]  <<  std::endl;
	std::cout << "[" << MV[4]  << " " << MV[5]  <<  " " << MV[6]  <<  " " << MV[7]  <<  std::endl;
	std::cout << "[" << MV[8]  << " " << MV[9]  <<  " " << MV[10] <<  " " << MV[11] <<  std::endl;
	std::cout << "[" << MV[12] << " " << MV[13] <<  " " << MV[14] <<  " " << MV[15] <<  std::endl;
	std::cout << "P:" << std::endl;
	std::cout << "[" << P[0]  << " " << P[1]  <<  " " << P[2]  <<  " " << P[3]  <<  std::endl;
	std::cout << "[" << P[4]  << " " << P[5]  <<  " " << P[6]  <<  " " << P[7]  <<  std::endl;
	std::cout << "[" << P[8]  << " " << P[9]  <<  " " << P[10] <<  " " << P[11] <<  std::endl;
	std::cout << "[" << P[12] << " " << P[13] <<  " " << P[14] <<  " " << P[15] <<  std::endl;
	*/
	// TODO: check
	//masses.render(camera.GetModelView(), camera.GetProjection());
	cubes.render(camera.GetModelView(), camera.GetProjection(), camera.GetNormal());

	glutSwapBuffers();
}

void Keys(unsigned char key, int x, int y)
{
	float px = 0.0, py = 0.0, pz = 0.0;
	//float lx = 0.0, ly = 0.0, lz = 0.0;
	if (key == 'q') px = 10.0;
	if (key == 'a') px = -10.0;
	if (key == 'w') py = 10.0;
	if (key == 's') py = -10.0;
	if (key == 'e') pz = 10.0;
	if (key == 'd') pz = -10.0;
	if (px != 0.0 || py != 0.0 || pz != 0.0) {
		//std::cout << "Move camera: " << px << ", " << py << ", " << pz \
		<< std::endl;
		unsigned int width = viewport.GetWidth();
		unsigned int height = viewport.GetHeight();
		masses.resizeWindow(
			viewport.GetNear(),
			viewport.GetFar(),
			viewport.GetFieldOfView(),
			glm::length(camera.GetLook()-camera.GetPosition()),
			cube_spacing*resolution_multiplier,
			width, height);
		glutPostRedisplay();
	}
	camera.MovePosition(px, py, pz);

	if (key == 'c')
		play = true;
	if (key == 'v')
		play = false;

	if (key == 'z') {
		Step();
		std::cout << "Stepping, frame: " << frame << std::endl;
		glutPostRedisplay();
	}

	if (key == 27) {
		//PrintMassesAndSprings();

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
		std::cerr << "Error: Failed to initialize GLEW." << std::endl;
		return 1;
	}
	std::cout << "Using GLEW " << glewGetString(GLEW_VERSION) \
	<< "." << std::endl;
	if (!GLEW_VERSION_4_2)
	{
		std::cerr << "Error: OpenGL 4.2 not supported." << std::endl;
		return 1;
	}
	std::cout << "Using OpenGL " << glGetString(GL_VERSION) << "." \
	<< std::endl;


	// set CUDA/OpenGL device
	std::cout << "Set CUDA/OpenGL device." << std::endl;
	SigAsiaDemo::setGLDevice(device);

	// load shaders
	std::cout << "Load shaders." << std::endl;
	masses.loadShaders();
	cubes.loadShaders();

	// load buffers
	std::cout << "Load buffers." << std::endl;
	masses.loadBuffers();

	// initialize OpenGL
	std::cout << "Initialize OpenGL." << std::endl;
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.7, 0.7, 1.0, 1.0);

	// resize viewport
	std::cout << "Resize viewport." << std::endl;
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());

	// fill masses
	std::cout << "Fill masses." << std::endl;

	springs.setConstants(10000.0, 1000.0);
	cubes.setConstants(10000.0, 1000.0);

	/*
	// TODO: replace by creators
	float offset = 20.0f;
	unsigned int m = 20;
	for (unsigned int i = 0; i < m; i++) {
		if (i == m-1) {
			masses.push(SigAsiaDemo::Mass(
				1.0,
				0.0, static_cast<float>(i*2) + offset, 0.0,
				0.0, 0.0, 0.0,
				0,
				1.0));
		} else {
			masses.push(SigAsiaDemo::Mass(
				1.0,
				0.0, static_cast<float>(i*2) + offset, 0.0,
				0.0, 0.0, 0.0,
				0,
				1.0));
		}
	}
	
	std::cout << "Fill springs." << std::endl;
	for (unsigned int i = 0; i < m-1; i++) {
		springs.push(SigAsiaDemo::Spring(masses, i, i+1, 10000.0, 1000.0));
	}
	*/

	std::cout << "Setup cube." << std::endl;

	cubes.push(SigAsiaDemo::Cube(
		-40.0, 20.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	/*
	cubes.push(SigAsiaDemo::Cube(
		-20.0, 30.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	*/
	cubes.push(SigAsiaDemo::Cube(
		0.0, 40.0, 0.0,		// position
		10, 10, 10,			// size
		cube_spacing*0.5,	// spacing
		4.0,				// mass
		cube_radius*0.5	// radius
		));
	/*
	cubes.push(SigAsiaDemo::Cube(
		20.0, 50.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	*/
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
	/*
	cubes.push(SigAsiaDemo::Cube(
		-20.0, 50.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	*/
	cubes.push(SigAsiaDemo::Cube(
		0.0, 60.0, 0.0,		// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	/*
	cubes.push(SigAsiaDemo::Cube(
		20.0, 70.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));
	*/
	cubes.push(SigAsiaDemo::Cube(
		40.0, 80.0, 0.0,	// position
		10, 10, 10,			// size
		cube_spacing,		// spacing
		4.0,				// mass
		cube_radius		// radius
		));

	std::cout << "Create cubes." << std::endl;
	cubes.create(masses, springs);

	std::cout << "Added " << masses.size() << " masses." << std::endl;
	std::cout << "Added " << springs.size() << " springs." << std::endl;
	std::cout << "Creation complete." << std::endl;

	std::cout << "Initialize masses." << std::endl;
	Step();
	std::cout << "Initialization complete." << std::endl;

	// register callbacks
	std::cout << "Register callbacks." << std::endl;
	glutIdleFunc(Idle);
	glutDisplayFunc(Render);
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(Keys);
	glutSpecialFunc(SpecialKeys);

	// enter GLUT event processing cycle
	std::cout << "Enter GLUT event processing cycle." << std::endl;
	glutMainLoop();

	return 0;
}
