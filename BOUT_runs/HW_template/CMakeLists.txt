cmake_minimum_required(VERSION 3.13)

project(hasegawa-wakatani LANGUAGES C CXX)

if(CMAKE_BINARY_DIR MATCHES "^${CMAKE_CURRENT_SOURCE_DIR}(/|$)")
  message(FATAL_ERROR "Build outside HW_template; use ../HW_N256/build_release")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development NumPy)

# Default to the sibling BOUT-dev checkout, independently of the build cwd.
# BOUT_ROOT may also be an installed prefix or a configured BOUT++ build tree.
get_filename_component(_default_bout_root
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../BOUT-dev" ABSOLUTE)
set(BOUT_ROOT "${_default_bout_root}" CACHE PATH
  "BOUT++ installation prefix, checkout, or configured build directory")

if (NOT TARGET bout++::bout++)
  find_package(bout++ CONFIG REQUIRED PATHS
    "${BOUT_ROOT}"
    "${BOUT_ROOT}/build_release"
    "${BOUT_ROOT}/build" NO_DEFAULT_PATH)
endif()

bout_add_example(hasegawa-wakatani SOURCES hw.cxx)

# Resolve the Python module directory relative to this model at configure time.
get_filename_component(HW_PYTHON_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
target_compile_definitions(hasegawa-wakatani PRIVATE
  HW_PYTHON_MODULE_DIR="${HW_PYTHON_MODULE_DIR}")

target_link_libraries(hasegawa-wakatani Python::Python Python::NumPy util)
