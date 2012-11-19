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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/massVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/massVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/massGS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/massGS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/massFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/massFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/planeVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/planeVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/planeFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/planeFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/screenVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/screenVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/screenFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/screenFS.glsl COPYONLY)

cuda_add_executable(SiggraphAsiaDemo main.cu viewport.cu mass.cu spring.cu device.cu camera.cu cube.cu)
target_link_libraries (SiggraphAsiaDemo ${LIBS})

