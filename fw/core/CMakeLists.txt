cmake_minimum_required(VERSION 3.16)
project(nf_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(NF_SANITIZE "Build with -fsanitize=address,undefined" OFF)
option(NF_BUILD_TESTS "Build unit tests" ON)

# MCU-friendly flags: no exceptions/RTTI, float32 discipline (-Wdouble-promotion).
set(NF_WARN -Wall -Wextra -Werror -Wshadow -Wdouble-promotion -fno-exceptions -fno-rtti)
if(NF_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

set(NF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
find_package(Python3 COMPONENTS Interpreter)

add_library(nf_core STATIC
  src/filters.cpp
  src/estimator.cpp
  src/control.cpp
  src/policy.cpp
  src/supervisor.cpp
  src/control_loop.cpp)
target_include_directories(nf_core PUBLIC include)
target_compile_options(nf_core PRIVATE ${NF_WARN})
set_target_properties(nf_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NF_BUILD_TESTS)
  enable_testing()
  set(NF_TESTS test_math test_filters test_estimator test_control test_policy test_supervisor test_control_loop)
  foreach(t ${NF_TESTS})
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE nf_core)
    target_compile_options(${t} PRIVATE ${NF_WARN})
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
  add_executable(bench_policy tests/bench_policy.cpp)
  target_link_libraries(bench_policy PRIVATE nf_core)
  target_compile_options(bench_policy PRIVATE ${NF_WARN})
  add_test(NAME bench_policy COMMAND bench_policy)
  # Bit-exact check of a generated policy (tools/codegen.py) against the Python
  # golden model's vectors (tools/gen_vectors.py). Driven by tools/pipeline.py:
  #   -DNF_GEN_POLICY_DIR=<dir with nf_policy_gen.{hpp,cpp}> -DNF_GEN_POLICY_VECTORS=<vectors.bin>
  set(NF_GEN_POLICY_DIR "" CACHE PATH "Directory with generated nf_policy_gen.hpp/.cpp")
  set(NF_GEN_POLICY_VECTORS "" CACHE FILEPATH "Golden vector file for test_policy_bitexact")
  if(NF_GEN_POLICY_DIR)
    add_executable(test_policy_bitexact ${NF_ROOT}/test/tools/cpp/test_policy_bitexact.cpp
                                        ${NF_GEN_POLICY_DIR}/nf_policy_gen.cpp)
    target_include_directories(test_policy_bitexact PRIVATE ${NF_GEN_POLICY_DIR})
    target_link_libraries(test_policy_bitexact PRIVATE nf_core)
    target_compile_options(test_policy_bitexact PRIVATE ${NF_WARN})
    if(NF_GEN_POLICY_VECTORS)
      add_test(NAME policy_bitexact COMMAND test_policy_bitexact ${NF_GEN_POLICY_VECTORS})
    endif()
  endif()
  if(Python3_Interpreter_FOUND)
    add_test(NAME vehicle_params_up_to_date
             COMMAND ${Python3_EXECUTABLE} ${NF_ROOT}/tools/gen_params.py --check)
  endif()
endif()
