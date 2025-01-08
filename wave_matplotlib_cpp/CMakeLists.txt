cmake_minimum_required(VERSION 3.10)
project(WavePlot)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python in the conda environment
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include Python3 and NumPy headers from the conda environment
include_directories( /home/ali/miniconda3/envs/cpp/include/python3.10/)                          # Python headers
include_directories(/home/ali/miniconda3/envs/cpp/lib/python3.10/site-packages/numpy/_core/include) # NumPy headers

# Include matplotlib-cpp
include_directories(~/libraries/matplotlib-cpp/)

# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# include_directories(${Python3_INCLUDE_DIRS})
# target_link_libraries(wave_plot ${Python3_LIBRARIES})

# Source directory
set(SOURCE_FILES src/wave_matplotlibcpp.cpp)

# Add the executable
add_executable(wave_plot ${SOURCE_FILES})





