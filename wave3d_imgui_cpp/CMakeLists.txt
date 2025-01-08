cmake_minimum_required(VERSION 3.10)
project(imgui_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set gcc version
set(CMAKE_C_COMPILER "/usr/bin/gcc-8")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
set(CUDA_HOST_COMPILER "/usr/bin/gcc-8")

# Include vcpkg toolchain
if (DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

#set SOURCE dir
set(SOURCES "src")

# Source files
add_executable(imgui_example ${SOURCES}/main.cpp)

cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

# Find necessary packages
find_package(glfw3 3.3 REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)
# find_package(GLUE REQUIRED)

# Add CUDA support
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
cuda_add_library(wave_cuda ${SOURCES}/wave_cuda.cu)

# Link CUDA library to the executable
target_link_libraries(imgui_example PUBLIC wave_cuda ${CUDA_LIBRARIES})






include_directories(
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/imgui
)
# Include and link
target_link_libraries(imgui_example PRIVATE
    glfw
    imgui::imgui
    OpenGL::GL
    GLEW::GLEW
    OpenGL::GLU
)

# Link CUDA library to the executable
target_link_libraries(imgui_example PUBLIC wave_cuda ${CUDA_LIBRARIES})