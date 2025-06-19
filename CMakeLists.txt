cmake_minimum_required(VERSION 3.18)
project(motion_decision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/absolute/path/to/libtorch")  # <-- change this!

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(motion_decision motion_decision.cpp)

target_link_libraries(motion_decision "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_include_directories(motion_decision PRIVATE ${OpenCV_INCLUDE_DIRS})
set_property(TARGET motion_decision PROPERTY CXX_STANDARD 17)

# Avoid runtime errors with Torch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

