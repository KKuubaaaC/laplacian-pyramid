# Public API tests — no headers from src/ 
add_executable(pyramid_tests_api laplacian_pyramid_test.cpp)

target_include_directories(pyramid_tests_api
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(pyramid_tests_api SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(pyramid_tests_api
    PRIVATE
        pyramid_lib
        GTest::gtest
        GTest::gtest_main
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

# Internal unit tests for modules under src/ (image_arithmetic, separable_filter)
add_executable(pyramid_tests_internal
    image_arithmetic_test.cpp
    separable_filter_test.cpp
)

target_include_directories(pyramid_tests_internal
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

target_include_directories(pyramid_tests_internal SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(pyramid_tests_internal
    PRIVATE
        pyramid_lib
        GTest::gtest
        GTest::gtest_main
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

include(GoogleTest)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    gtest_discover_tests(pyramid_tests_api DISCOVERY_MODE PRE_TEST)
    gtest_discover_tests(pyramid_tests_internal DISCOVERY_MODE PRE_TEST)
else()
    gtest_discover_tests(pyramid_tests_api)
    gtest_discover_tests(pyramid_tests_internal)
endif()
