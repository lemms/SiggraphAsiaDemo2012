# This file is part of SigAsiaDemo2012.

# SigAsiaDemo2012 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SigAsiaDemo2012 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SigAsiaDemo2012.  If not, see <http://www.gnu.org/licenses/>.

# Copyright 2012 Laurence Emms

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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubeVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/cubeVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cubeFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/cubeFS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pointVS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/pointVS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pointGS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/pointGS.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pointFS.glsl
	${CMAKE_CURRENT_BINARY_DIR}/pointFS.glsl COPYONLY)

cuda_add_executable(SiggraphAsiaDemo main.cu viewport.cu mass.cu spring.cu device.cu camera.cu cube.cu glew.c)
target_link_libraries (SiggraphAsiaDemo ${LIBS})

