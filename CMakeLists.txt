cmake_minimum_required(VERSION 3.20)
project(office_ai)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(program
    main.cpp
    ${SOURCES}
)

target_include_directories(program PRIVATE include)