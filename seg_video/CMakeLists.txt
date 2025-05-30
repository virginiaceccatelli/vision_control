cmake_minimum_required(VERSION 3.10)
project(GroundPlaneDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/opt/homebrew/Cellar/pytorch")  

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(seg_video transpose.cpp)

target_include_directories(ground_plane_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ground_plane_app "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

set_target_properties(ground_plane_app PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

