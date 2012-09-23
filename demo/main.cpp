/*
Siggraph Asia 2012 Demo

Laurence Emms
*/

#include <iostream>
using namespace std;

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>
#ifdef FREEGLUT
	#include <GL/freeglut_ext.h>
#endif

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i) {
	}
	cout << "Siggraph Asia 2012 Demo" << endl;

	// setup glut
	glutInit(&argc, argv);
#ifdef FREEGLUT
	cout << "FreeGLUT" << endl;
	if (glutGet(GLUT_VERSION) < 20001) {
	    cout << "Sorry, you need freeglut version 2.0.1 or later to run this program." << endl;
	    return 1;
	}
#else
	cout << "GLUT" << endl;
#endif
	glutInitDisplayMode(GLUT_DEPTH | 
			GLUT_DOUBLE | 
			GLUT_RGBA | 
			GLUT_MULTISAMPLE);
	glutInitWindowPosition(0, 0);

	return 0;
}
