set(PROJ_SRC
    main.cpp
)

find_package (Threads REQUIRED)
find_package (OpenCV REQUIRED)

add_executable(calib ${PROJ_SRC})
target_link_libraries(calib ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
