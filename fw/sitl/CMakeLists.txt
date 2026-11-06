# SITL bridge: the firmware core (fw/core) as a host shared library with a C ABI,
# driven from Python (fw/sitl/sitl.py) via ctypes.
#
#   cmake -S fw/sitl -B fw/sitl/build && cmake --build fw/sitl/build -j
#
# Produces fw/sitl/build/libnf_sitl.so.
cmake_minimum_required(VERSION 3.16)
project(nf_sitl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build the unmodified firmware core (no unit tests) as a PIC static lib.
set(NF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_CURRENT_BINARY_DIR}/nf_core)

add_library(nf_sitl SHARED nf_sitl.cpp)
target_link_libraries(nf_sitl PRIVATE nf_core)
target_include_directories(nf_sitl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The bridge is host-only code (heap, std::vector are fine here); keep the
# core's warning discipline otherwise.
target_compile_options(nf_sitl PRIVATE -Wall -Wextra -Werror -Wshadow -Wdouble-promotion)
set_target_properties(nf_sitl PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden
                                          LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
