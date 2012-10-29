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

// globals
SigAsiaDemo::Viewport viewport;
SigAsiaDemo::Camera camera;
SigAsiaDemo::MassList masses;
SigAsiaDemo::SpringList springs;

size_t frame = 0;
float dt = 0.01;

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
	}
}

void Step()
{
	//std::cout << "Frame: " << frame << std::endl;
	//std::cout << "upload masses." << std::endl;
	masses.upload();
	//std::cout << "upload springs." << std::endl;
	springs.upload(masses);
	masses.startFrame();

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK1(dt);

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK2(dt);

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK3(dt);

	masses.clearForces();
	springs.applySpringForces(masses);
	masses.evaluateK4(dt);

	masses.update(dt);

	//masses.download();
	//springs.download();
}

void Idle()
{
	// TODO: come up with a better metric
	//if (frame % 100 == 0)
		//glutPostRedisplay();

	//Step();

	// TODO: remove
	/*
	if (frame == 1000) {
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
				std::cout << "velocity k1: (";
				std::cout << mass0->_k1x << ", ";
				std::cout << mass0->_k1y << ", ";
				std::cout << mass0->_k1z << ")" << std::endl;
				std::cout << "velocity k2: (";
				std::cout << mass0->_k2x << ", ";
				std::cout << mass0->_k2y << ", ";
				std::cout << mass0->_k2z << ")" << std::endl;
				std::cout << "velocity k3: (";
				std::cout << mass0->_k3x << ", ";
				std::cout << mass0->_k3y << ", ";
				std::cout << mass0->_k3z << ")" << std::endl;
				std::cout << "velocity k4: (";
				std::cout << mass0->_k4x << ", ";
				std::cout << mass0->_k4y << ", ";
				std::cout << mass0->_k4z << ")" << std::endl;
				std::cout <<std::endl;
			}
		}

		springs.download();

		for (size_t i = 0; i < springs.size(); i++) {
			SigAsiaDemo::Spring *spring0 = springs.getSpring(i);
			if (spring0) {
				std::cout << "Spring " << i << std::endl;
				std::cout << "mass 0: " << spring0->_mass0 << std::endl;
				std::cout << "mass 1: " << spring0->_mass1 << std::endl;
				std::cout << "ks: " << spring0->_ks << std::endl;
				std::cout << "kd: " << spring0->_kd << std::endl;
				std::cout << "l0: " << spring0->_l0 << std::endl;
				std::cout << "f0: (";
				std::cout << spring0->_fx0 << ", ";
				std::cout << spring0->_fy0 << ", ";
				std::cout << spring0->_fz0 << ")" << std::endl;
				std::cout << "f1: (";
				std::cout << spring0->_fx1 << ", ";
				std::cout << spring0->_fy1 << ", ";
				std::cout << spring0->_fz1 << ")" << std::endl;
				std::cout <<std::endl;
			}
		}

		exit(0);
	}
	*/

	//frame++;
}

void Reshape(int width, int height)
{
	viewport.ResizeWindow(width, height);
	camera.ResizeWindow(width, height);
}

void Render()
{
	std::cout << "Render" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
	masses.render(camera.GetModelView(), camera.GetProjection());

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
		std::cout << "Move camera: " << px << ", " << py << ", " << pz \
		<< std::endl;
		glutPostRedisplay();
	}
	camera.MovePosition(px, py, pz);

	if (key == 'i') {
		frame++;
		Step();
		glutPostRedisplay();
	}

	if (key == 27) {
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
				std::cout << "velocity k1: (";
				std::cout << mass0->_k1x << ", ";
				std::cout << mass0->_k1y << ", ";
				std::cout << mass0->_k1z << ")" << std::endl;
				std::cout << "velocity k2: (";
				std::cout << mass0->_k2x << ", ";
				std::cout << mass0->_k2y << ", ";
				std::cout << mass0->_k2z << ")" << std::endl;
				std::cout << "velocity k3: (";
				std::cout << mass0->_k3x << ", ";
				std::cout << mass0->_k3y << ", ";
				std::cout << mass0->_k3z << ")" << std::endl;
				std::cout << "velocity k4: (";
				std::cout << mass0->_k4x << ", ";
				std::cout << mass0->_k4y << ", ";
				std::cout << mass0->_k4z << ")" << std::endl;
				std::cout <<std::endl;
			}
		}

		springs.download();

		for (size_t i = 0; i < springs.size(); i++) {
			SigAsiaDemo::Spring *spring0 = springs.getSpring(i);
			if (spring0) {
				std::cout << "Spring " << i << std::endl;
				std::cout << "mass 0: " << spring0->_mass0 << std::endl;
				std::cout << "mass 1: " << spring0->_mass1 << std::endl;
				std::cout << "ks: " << spring0->_ks << std::endl;
				std::cout << "kd: " << spring0->_kd << std::endl;
				std::cout << "l0: " << spring0->_l0 << std::endl;
				std::cout << "f0: (";
				std::cout << spring0->_fx0 << ", ";
				std::cout << spring0->_fy0 << ", ";
				std::cout << spring0->_fz0 << ")" << std::endl;
				std::cout << "f1: (";
				std::cout << spring0->_fx1 << ", ";
				std::cout << spring0->_fy1 << ", ";
				std::cout << spring0->_fz1 << ")" << std::endl;
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

	// initialize OpenGL
	std::cout << "Initialize OpenGL." << std::endl;
	glClearColor(0.0, 0.0, 0.0, 0.0);

	// resize viewport
	std::cout << "Resize viewport." << std::endl;
	glViewport(0, 0, viewport.GetWidth(), viewport.GetHeight());

	// TODO: replace by creators
	// fill masses
	std::cout << "Fill masses." << std::endl;
	for (unsigned int i = 0; i < 100; i++)
		masses.push(SigAsiaDemo::Mass(
			1.0,
			0.0, static_cast<float>(i*2), 0.0,
			0.0, 0.0, 0.0,
			0,
			1.0));
	
	std::cout << "Fill springs." << std::endl;
	for (unsigned int i = 0; i < 99; i++)
		springs.push(SigAsiaDemo::Spring(masses, i, i+1));

	std::cout << "Initialize masses." << std::endl;
	Step();

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
