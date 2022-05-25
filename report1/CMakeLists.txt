cmake_minimum_required(VERSION 3.20)
project(numopt_report1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(numopt_report1
    main.cpp
)

target_link_libraries(numopt_report1 PUBLIC
    Eigen3::Eigen
)
