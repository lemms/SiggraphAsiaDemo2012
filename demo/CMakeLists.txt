set(DIRS "")
set(LIBS "")

# add CUDA support
find_package (CUDA)

# add OpenGL support
find_package (OpenGL)
if (OPENGL_FOUND)
	set(DIRS ${DIRS} ${OPENGL_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${OPENGL_LIBRARIES})
	message(STATUS " OPENGL_INCLUDE_DIR:  ${OPENGL_INCLUDE_DIR}")
	message(STATUS " OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")
endif (OPENGL_FOUND)

# add GLEW support
find_package (GLEW)
if (GLEW_FOUND)
	set(DIRS ${DIRS} ${GLEW_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${GLEW_LIBRARIES})
	message(STATUS " GLEW_INCLUDE_DIR:  ${GLEW_INCLUDE_DIR}")
	message(STATUS " GLEW_LIBRARIES:  ${GLEW_LIBRARIES}")
endif (GLEW_FOUND)

# add GLUT support
find_package (GLUT)
if (GLUT_FOUND)
	set(DIRS ${DIRS} ${GLUT_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${GLUT_LIBRARIES})
	message(STATUS " GLUT_INCLUDE_DIR:  ${GLUT_INCLUDE_DIR}")
	message(STATUS " GLUT_LIBRARIES: ${GLUT_LIBRARIES}")
endif (GLUT_FOUND)

include_directories(${DIRS})

cuda_add_executable(SiggraphAsiaDemo main.cu viewport.cu mass.cu spring.cu)
target_link_libraries (SiggraphAsiaDemo ${LIBS})

