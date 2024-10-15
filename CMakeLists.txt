cmake_minimum_required(VERSION 3.28)
project(CuPAAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(CuPAAL main.cpp)
add_library(baume src/cupaal/baum.cpp src/cupaal/baum.h)

set(storm_DIR /opt/storm/build)
find_package(storm REQUIRED)

set(HEADER_FILES "")

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_FILES} ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers baume)

target_link_directories(CuPAAL PUBLIC /src/)
target_include_directories(baume PUBLIC ${HEADER_FILES} ${storm_INCLUDE_DIR} ${storm-parsers_INCLUDE_DIR})
target_link_libraries(baume PRIVATE storm storm-parsers)