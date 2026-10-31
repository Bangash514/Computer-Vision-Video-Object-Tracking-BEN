cmake_minimum_required(VERSION 3.10)
project(VideoObjectTracking CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(make_synthetic_video cpp/make_synthetic_video.cpp)
target_include_directories(make_synthetic_video PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(make_synthetic_video ${OpenCV_LIBS})

add_executable(track_video cpp/track_video.cpp)
target_include_directories(track_video PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(track_video ${OpenCV_LIBS})
