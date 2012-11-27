set(DIRS "")
set(LIBS "")

# add CUDA support
find_package (CUDA)

# add OpenGL support
find_package (OpenGL)
if (OPENGL_FOUND)
	set(DIRS ${DIRS} ${OPENGL_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${OPENGL_LIBRARIES})
	message(STATUS " OPENGL_INCLUDE_DIR:  ${OPENGL_INCLUDE_DIRS}")
	message(STATUS " OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")
endif (OPENGL_FOUND)

# add GLEW support
find_package (GLEW)
if (GLEW_FOUND)
	set(DIRS ${DIRS} ${GLEW_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${GLEW_LIBRARIES})
	message(STATUS " GLEW_INCLUDE_DIR:  ${GLEW_INCLUDE_DIRS}")
	message(STATUS " GLEW_LIBRARIES:  ${GLEW_LIBRARIES}")	
	ADD_DEFINITIONS(-DBUILD_VTI -DGLEW_STATIC)
endif (GLEW_FOUND)

# add GLUT support
ADD_DEFINITIONS(-DBUILD_VTI -DFREEGLUT_STATIC)
find_package (GLUT)
if (GLUT_FOUND)
	set(DIRS ${DIRS} ${GLUT_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${GLUT_LIBRARIES})
	message(STATUS " GLUT_INCLUDE_DIR:  ${GLUT_INCLUDE_DIRS}")
	message(STATUS " GLUT_LIBRARIES: ${GLUT_LIBRARIES}")
endif (GLUT_FOUND)

include_directories(${DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/depthVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/depthVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/depthGS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/depthGS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/depthFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/depthFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/overlapVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/overlapVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/overlapGS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/overlapGS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/overlapFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/overlapFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/weightVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/weightVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/weightGS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/weightGS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/weightFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/weightFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outlineVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/outlineVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/outlineFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/outlineFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/planeVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/planeVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/planeFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/planeFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/screenVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/screenVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/screenFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/screenFS.glsl COPYONLY)

cuda_add_executable(SiggraphAsiaDemo main.cu viewport.cu mass.cu spring.cu device.cu camera.cu cube.cu glew.c)
target_link_libraries (SiggraphAsiaDemo ${LIBS})

