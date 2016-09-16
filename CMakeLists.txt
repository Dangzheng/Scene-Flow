#Project
cmake_minimum_required(VERSION 2.4)
project(flow)

#Flags
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g -ggdb -msse4.2 -stdlib=libstdc++")
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libstdc++")
add_compile_options(-std=c++11)

# Flow
find_package(OpenCV REQUIRED)
file(GLOB FLOW_SRC_FILES "*.cpp")
add_executable(flow ${FLOW_SRC_FILES})
#add_executable(flow flow)

target_link_libraries(flow ${OpenCV_LIBS} png)


