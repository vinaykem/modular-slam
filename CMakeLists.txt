cmake_minimum_required(VERSION 3.10)

# set the project name
project(VisualOdometry)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(matrix)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add the executable
add_executable(vo vo.cpp)
target_link_libraries(vo PUBLIC matrix ${OpenCV_LIBS})

target_include_directories(vo PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/matrix")
configure_file("${PROJECT_SOURCE_DIR}/Rt_Corrected.txt" "${PROJECT_BINARY_DIR}" COPYONLY)
