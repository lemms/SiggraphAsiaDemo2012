# Copyright (c) 1998, Regents of the University of California
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# Neither the name of the University of California, Berkeley nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
#
IF (WIN32)
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
${GLEW_ROOT_DIR}/include
DOC "The directory where GL/glew.h resides")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
set(GLEWNAMES glew GLEW glew64 glew64s)
else ()
set(GLEWNAMES glew GLEW glew32 glew32s)
endif (CMAKE_SIZEOF_VOID_P EQUAL 8)

FIND_LIBRARY( GLEW_LIBRARY
NAMES ${GLEWNAMES}
PATHS
${GLEW_ROOT_DIR}/bin
${GLEW_ROOT_DIR}/lib
DOC "The GLEW library")
ELSE (WIN32)
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
/usr/include
/usr/local/include
/sw/include
/opt/local/include
${GLEW_ROOT_DIR}/include
DOC "The directory where GL/glew.h resides")
FIND_LIBRARY( GLEW_LIBRARY
NAMES GLEW libGLEW
PATHS
/usr/lib64
/usr/lib
/usr/local/lib64
/usr/local/lib
/sw/lib
/opt/local/lib
${GLEW_ROOT_DIR}/lib
DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)
SET( FOUND_GLEW 1)
ELSE (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)
SET( FOUND_GLEW 0)
ENDIF (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)

MARK_AS_ADVANCED( FOUND_GLEW )
