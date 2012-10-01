# The MIT License (MIT)
# 
# Copyright (c) 2007 NVIDIA
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# - Try to find GLEW
# Once done this will define
#  GLEW_FOUND - System has GLEW
#  GLEW_INCLUDE_DIRS - The GLEW include directories
#  GLEW_LIBRARIES - The libraries needed to use GLEW
#  GLEW_DEFINITIONS - Compiler switches required for using GLEW

find_package(PkgConfig)
pkg_check_modules(PC_GLEW QUIET glew)
set(GLEW_DEFINITIONS ${PC_GLEW_CFLAGS_OTHER})

find_path(GLEW_INCLUDE_DIR GL/glew.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		HINTS ${PC_GLEW_INCLUDEDIR} ${PC_GLEW_INCLUDE_DIRS}
		PATH_SUFFIXES glew)

find_library(GLEW_LIBRARY 
		NAMES glew GLEW libglew glew32 libglew32 glew32s
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/usr/bin
		/usr/local/bin
		/sw/lib
		/opt/local/lib
		HINTS ${PC_GLEW_LIBDIR} ${PC_GLEW_LIBRARY_DIRS})

set(GLEW_LIBRARIES ${GLEW_LIBRARY})
set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set GLEW_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GLEW  DEFAULT_MSG
                                  GLEW_LIBRARY GLEW_INCLUDE_DIR)

mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARY)

