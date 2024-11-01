cmake_minimum_required(VERSION 3.28)
project(CuPAAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(CuPAAL main.cpp)

set(storm_DIR /opt/storm/build)
set(cudd_DIR ${storm_DIR}/resources/3rdparty/cudd-3.0.0/src/cudd3-build)
set(cudd_src_DIR /opt/storm/resources/3rdparty/cudd-3.0.0)

find_package(storm REQUIRED)

set(HEADER_FILES "")

target_include_directories(${PROJECT_NAME} PUBLIC
        ${HEADER_FILES}
        ${storm_INCLUDE_DIR}
        ${storm-parsers_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} PRIVATE storm storm-parsers baum)

add_library(baum src/cupaal/baum.cpp src/cupaal/baum.h
        src/cupaal/Cudd_extensions.cpp
        src/cupaal/Cudd_extensions.h)

target_link_directories(CuPAAL PUBLIC /src/)
target_include_directories(baum PUBLIC
        ${HEADER_FILES}
        ${storm_INCLUDE_DIR}
        ${storm-parsers_INCLUDE_DIR}
        ${cudd_src_DIR}/cudd
        ${cudd_DIR} # Includes config.h for cuddInt.h
        ${cudd_src_DIR}/st
        ${cudd_src_DIR}/epd
)
target_link_libraries(baum PRIVATE storm storm-parsers)
