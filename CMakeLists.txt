#Project
cmake_minimum_required(VERSION 3.6)
project(flow)

#Flags
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g -ggdb -msse4.2 -stdlib=libc++")
# set(CMAKE_BUILD_TYPE Debug)
if (APPLE)
   set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
   set(CMAKE_C_COMPILER "/usr/bin/clang")
   set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libc++")
   add_compile_options(-std=c++11)
endif (APPLE)
# Flow
find_package(OpenCV REQUIRED)
file(GLOB FLOW_SRC_FILES "*.cpp")
add_executable(flow ${FLOW_SRC_FILES})

target_link_libraries(flow ${OpenCV_LIBS} png)

# usage
# cmake -D CMAKE_PREFIX_PATH=/Users/Zheng/opencv3 .

