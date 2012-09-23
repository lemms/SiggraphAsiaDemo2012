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
#include <GL/glut.h>
#ifdef FREEGLUT
	#include <GL/freeglut_ext.h>
#endif

// globals
unsigned int window_width = 1024;
unsigned int window_height = 768;

void parseArgs(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i) {
		stringstream stream(argv[i]);
		if (stream.str() == "-w") {
			if (i++ > argc)
				return;
			stream.str(argv[i]);
			std::cout << stream.str() << std::endl;
			stream >> window_width;
		}
		if (stream.str() == "-h") {
			if (i++ > argc)
				return;
			stream.str(argv[i]);
			std::cout << stream.str() << std::endl;
			stream >> window_height;
		}
	}
}

void idle()
{
	glutPostRedisplay();
}

void render()
{
	glutSwapBuffers();
}

int main(int argc, char **argv)
{
	parseArgs(argc, argv);
	cout << "Siggraph Asia 2012 Demo" << endl;
	cout << "Laurence Emms" << endl;

	// setup glut
	glutInit(&argc, argv);
#ifdef FREEGLUT
	cout << "using FreeGLUT" << endl;
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
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Siggraph Asia 2012 Mass Spring Demo");

	// register callbacks
	glutIdleFunc(idle);
	glutDisplayFunc(render);

	// enter GLUT event processing cycle
	glutMainLoop();

	return 0;
}
