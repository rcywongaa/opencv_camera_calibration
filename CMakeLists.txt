cmake_minimum_required(VERSION 3.0.0)
project(calib)
set(CMAKE_BUILD_TYPE Debug)
include_directories(include)

set(PROJ_SRC
    src/main.cpp
)

find_package (Threads REQUIRED)
find_package (OpenCV REQUIRED)

add_executable(calib ${PROJ_SRC})
target_link_libraries(calib ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
