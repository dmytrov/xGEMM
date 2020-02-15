cmake_minimum_required(VERSION 3.10)
project(xGEMM)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

if(CMAKE_COMPILER_IS_GNUCXX)

    #enable C++11 on older versions of cmake
    if (CMAKE_VERSION VERSION_LESS "3.1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()

    #enable warnings in Debug configurations
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    	add_compile_options(-Wall -Wpedantic)
    endif()

    #symbols are only exported from libraries/modules explicitly
    add_compile_options(-fvisibility=hidden)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

    #default SIMD configuration uses native build flags
    #when packaging and x86, use sse3 so the binaries work across multiple x86 variants
    if(NOT DEFAULT_SIMD_FLAGS)
        set(DEFAULT_SIMD_FLAGS "native")
    endif()
    if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr" AND X86)
        set(DEFAULT_SIMD_FLAGS "SSE3")
    endif()

    SET(ENABLE_SIMD_FLAGS "${DEFAULT_SIMD_FLAGS}" CACHE STRING "Set compiler SIMD flags")
    SET_PROPERTY(CACHE ENABLE_SIMD_FLAGS PROPERTY STRINGS none native SSE3 SSE4.2 AVX AVX2) 

    if(${ENABLE_SIMD_FLAGS} MATCHES "native")
        add_definitions(-march=native)
        if(X86)
            add_definitions(-mfpmath=both)
        endif()
        message(STATUS "Enabling native optimizations")
    elseif(${ENABLE_SIMD_FLAGS} MATCHES "AVX2")
        add_definitions(-mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mavx -mfma -mavx2 -mfpmath=both)
        message(STATUS "Enabling AVX2 instructions")
    elseif(${ENABLE_SIMD_FLAGS} MATCHES "AVX")
        add_definitions(-mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mavx -mfpmath=both)
        message(STATUS "Enabling AVX instructions")
    elseif(${ENABLE_SIMD_FLAGS} MATCHES "SSE4.2")
        add_definitions(-mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mfpmath=both)
        message(STATUS "Enabling SSE4.2 instructions")
    elseif(${ENABLE_SIMD_FLAGS} MATCHES "SSE3")
        add_definitions(-mmmx -msse -msse2 -msse3 -mfpmath=both)
        message(STATUS "Enabling SSE3 instructions")
    endif()

endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost)
find_package (Threads)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_subdirectory(src)
endif()
