# - Try to find FreeGLUT
# Once done this will define
#  FREEGLUT_FOUND - System has FreeGLUT
#  FREEGLUT_INCLUDE_DIRS - The FreeGLUT include directories
#  FREEGLUT_LIBRARIES - The libraries needed to use FreeGLUT
#  FREEGLUT_DEFINITIONS - Compiler switches required for using FreeGLUT


FIND_PATH(FREEGLUT_INCLUDE_DIR NAMES GL/freeglut.h)
FIND_LIBRARY(FREEGLUT_LIBRARY NAMES freeglut freeglut_static)

set(FREEGLUT_LIBRARIES ${FREEGLUT_LIBRARY})
set(FREEGLUT_INCLUDE_DIRS ${FREEGLUT_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FREEGLUT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(FreeGLUT  DEFAULT_MSG
                                  FREEGLUT_LIBRARY FREEGLUT_INCLUDE_DIR)

mark_as_advanced(FREEGLUT_INCLUDE_DIR FREEGLUT_LIBRARY)
