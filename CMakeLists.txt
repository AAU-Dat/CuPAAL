cmake_minimum_required(VERSION 3.28)
project(CuPAAL VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME} main.cpp)

set(storm_DIR /opt/storm/build)
find_package(storm REQUIRED)

set(HEADER_FILES "")

target_include_directories(${PROJECT_NAME} PUBLIC ${HEADER_FILES} ${storm_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers BaumWelch CuddExtensions Helpers)
target_link_directories(${PROJECT_NAME} PUBLIC /src/)

add_library(BaumWelch src/cupaal/baum_welch.cpp src/cupaal/baum_welch.h)

add_library(CuddExtensions src/cupaal/cudd_extensions.cpp src/cupaal/cudd_extensions.h)
target_include_directories(CuddExtensions
        PUBLIC
        /opt/storm/build/resources/3rdparty/cudd-3.0.0/include
        /opt/storm/resources/3rdparty/cudd-3.0.0/cudd
        /opt/storm/build/resources/3rdparty/cudd-3.0.0/src/cudd3-build
        /opt/storm/resources/3rdparty/cudd-3.0.0/st
        /opt/storm/resources/3rdparty/cudd-3.0.0/epd
        /opt/storm/resources/3rdparty/cudd-3.0.0/util
)

add_library(Helpers src/cupaal/helpers.cpp src/cupaal/helpers.h)
target_include_directories(Helpers PUBLIC ${HEADER_FILES} ${storm_INCLUDE_DIR})
target_link_libraries(Helpers PRIVATE storm storm-parsers)