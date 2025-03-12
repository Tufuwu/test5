#-------------------------------------------------------------------------------
#   wasisdk.toolchain.cmake
#   Fips cmake toolchain file for cross-compiling with the wasi-sdk.
#
#   The following cmake defines must be passed on the command line:
#
#   WASISDK_ROOT:   absolute path to the wasi-sdk directory
#-------------------------------------------------------------------------------

set(FIPS_PLATFORM WASISDK)
set(FIPS_PLATFORM_NAME "wasisdk")
set(FIPS_WASISDK 1)
set(FIPS_POSIX 1)

set(WASISDK_SYSROOT "${WASISDK_ROOT}/share/wasi-sysroot")

set(FIPS_WASISDK_POSTFIX ".wasm" CACHE STRING "wasisdk output postfix")

set(WASISDK_COMMON_FLAGS)
set(WASISDK_COMMON_FLAGS_RELEASE)
set(WASISDK_CXX_FLAGS)
set(WASISDK_LINKER_FLAGS)
set(WASISDK_LINKER_FLAGS_RELEASE)
set(WASISDK_EXE_LINKER_FLAGS)

if (NOT FIPS_EXCEPTIONS)
    set(WASISDK_CXX_FLAGS "${WASISDK_CXX_FLAGS} -fno-exceptions")
endif()

if (NOT FIPS_RTTI)
    set(WASISDK_CXX_FLAGS "${WASISDK_CXX_FLAGS} -fno-rtti")
endif()

set(WASISDK_OPT "-O3")

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)
set(CMAKE_CONFIGURATION_TYPES Debug Release Profiling)
set(CMAKE_C_COMPILER_TARGET "wasm32-wasi")
set(CMAKE_CXX_COMPILER_TARGET "wasm32-wasi")
set(CMAKE_SYSROOT "${WASISDK_SYSROOT}")
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_CXX_COMPILER_WORKS TRUE)

if(WIN32)
    set(WASI_HOST_EXE_SUFFIX ".exe")
else()
    set(WASI_HOST_EXE_SUFFIX "")
endif()
# specify cross-compilers
set(CMAKE_C_COMPILER "${WASISDK_ROOT}/bin/clang${WASI_HOST_EXE_SUFFIX}" CACHE PATH "gcc" FORCE)
set(CMAKE_CXX_COMPILER "${WASISDK_ROOT}/bin/clang++${WASI_HOST_EXE_SUFFIX}" CACHE PATH "g++" FORCE)
set(CMAKE_AR "${WASISDK_ROOT}/bin/ar${WASI_HOST_EXE_SUFFIX}" CACHE PATH "archive" FORCE)
set(CMAKE_LINKER "${WASISDK_ROOT}/bin/clang${WASI_HOST_EXE_SUFFIX}" CACHE PATH "linker" FORCE)
set(CMAKE_RANLIB "${WASISDK_ROOT}/bin/ranlib${WASI_HOST_EXE_SUFFIX}" CACHE PATH "ranlib" FORCE)

# only search for libraries and includes in the toolchain
#set(CMAKE_FIND_ROOT_PATH ${WASISDK_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_LIBRARIES 1)
set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
set(CMAKE_C_USE_RESPONSE_FILE_FOR_INCLUDES 1)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES 1)

set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")

set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> rc <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> rc <TARGET> <LINK_FLAGS> <OBJECTS>")

# c++ compiler flags
set(CMAKE_CXX_FLAGS "${WASISDK_COMMON_FLAGS} ${WASISDK_CXX_FLAGS} -fstrict-aliasing -Wall -Wno-multichar -Wextra -Wno-unknown-pragmas -Wno-ignored-qualifiers -Wno-long-long -Wno-overloaded-virtual -Wno-deprecated-writable-strings -Wno-unused-volatile-lvalue -Wno-inconsistent-missing-override -Wno-expansion-to-defined")
set(CMAKE_CXX_FLAGS_RELEASE "${WASISDK_COMMON_FLAGS_RELEASE} ${WASISDK_OPT} -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -D_DEBUG_ -D_DEBUG -DFIPS_DEBUG=1")
set(CMAKE_CXX_FLAGS_PROFILING "${WASISDK_OPT} -DNDEBUG --profiling")

# c compiler flags
set(CMAKE_C_FLAGS "${WASISDK_COMMON_FLAGS} -fstrict-aliasing -Wall -Wextra -Wno-multichar -Wno-unknown-pragmas -Wno-ignored-qualifiers -Wno-long-long -Wno-overloaded-virtual -Wno-deprecated-writable-strings -Wno-unused-volatile-lvalue -Wno-expansion-to-defined")
set(CMAKE_C_FLAGS_RELEASE "${WASISDK_OPT} ${WASISDK_COMMON_FLAGS_RELEASE} -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -D_DEBUG_ -D_DEBUG -DFIPS_DEBUG=1")
set(CMAKE_C_FLAGS_PROFILING "${WASISDK_OPT} -DNDEBUG --profiling")

# linker flags
set(CMAKE_EXE_LINKER_FLAGS "${WASISDK_COMMON_FLAGS} ${WASISDK_LINKER_FLAGS} ${WASISDK_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${WASISDK_OPT} ${WASISDK_COMMON_FLAGS_RELEASE} ${WASISDK_LINKER_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-O0 -g")
set(CMAKE_EXE_LINKER_FLAGS_PROFILING "--profiling ${WASISDK_OPT} ${WASISDK_LINKER_FLAGS_RELEASE}")

# dynamic lib linker flags
set(CMAKE_SHARED_LINKER_FLAGS "-shared ${WASISDK_COMMON_FLAGS} ${WASISDK_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${WASISDK_OPT} ${WASISDK_COMMON_FLAGS_RELEASE} ${WASISDK_LINKER_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${WASISDK_OPT} -g")
set(CMAKE_SHARED_LINKER_FLAGS_PROFILING "--profiling ${WASISDK_OPT} ${WASISDK_LINKER_FLAGS_RELEASE}")

# update cache variables for cmake gui
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Config Type" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "Generic C++ Compiler Flags" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" CACHE STRING "C++ Debug Compiler Flags" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "C++ Release Compiler Flags" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "Generic C Compiler Flags" FORCE)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}" CACHE STRING "C Debug Compiler Flags" FORCE)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "C Release Compiler Flags" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "Generic Linker Flags" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING "Debug Linker Flags" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "Release Linker Flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING "Generic Shared Linker Flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" CACHE STRING "Debug Shared Linker Flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "Release Shared Linker Flags" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS}" CACHE STRING "Static Lib Flags" FORCE)

# set the build type to use
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Compile Type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release Profiling)
