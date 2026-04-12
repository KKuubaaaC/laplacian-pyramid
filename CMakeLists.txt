cmake_minimum_required(VERSION 3.14)
project(LaplacianPyramid CXX)

option(ENABLE_CLANG_TIDY "Run clang-tidy on targets when compiling" OFF)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-extra-arg=-Wno-unknown-warning-option")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Werror
)

# AddressSanitizer + UndefinedBehaviorSanitizer in Debug 
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        $<$<CONFIG:Debug>:-g>
    )
    add_link_options(
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

find_package(GTest QUIET CONFIG)
if(NOT GTest_FOUND)
    find_package(GTest QUIET MODULE)
endif()
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# --- pyramid static library ---
add_library(pyramid_lib STATIC
    src/image_arithmetic.cpp
    src/separable_filter.cpp
    src/laplacian_pyramid.cpp
    src/pyramid_blend.cpp
    src/pyramid_utils.cpp
)

target_include_directories(pyramid_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_include_directories(pyramid_lib SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(pyramid_lib
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgproc
)

# --- demo executable ---
add_executable(pyramid_demo src/main.cpp)

target_include_directories(pyramid_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(pyramid_demo SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(pyramid_demo
    PRIVATE
        pyramid_lib
        opencv_imgproc
        opencv_imgcodecs
)

# --- tests ---
enable_testing()
add_subdirectory(tests)
