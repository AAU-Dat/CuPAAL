cmake_minimum_required(VERSION 3.28)
project(CuPAAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(CuPAAL main.cpp)

set(storm_DIR /opt/storm/build)
find_package(storm REQUIRED)

set(HEADER_FILES "")

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_FILES} ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers)