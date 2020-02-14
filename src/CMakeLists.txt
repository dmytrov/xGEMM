include_directories(${Boost_INCLUDE_DIRS})
include_directories(${xGEMM_SOURCE_DIR}/inc)

option(ENABLE_MMX "Enables MMX optimizations on x86/x86_64 targets." ON)
option(ENABLE_SSE "Enables SSE optimizations on x86/x86_64 targets." ON)
option(ENABLE_SSE2 "Enables SSE2 optimizations on x86/x86_64 targets." ON)
option(ENABLE_SSE3 "Enables SSE3 optimizations on x86/x86_64 targets." ON)
option(ENABLE_SSSE3 "Enables SSSE3 optimizations on x86/x86_64 targets." ON)
option(ENABLE_SSE4_1 "Enables SSE4_1 optimizations on x86/x86_64 targets." ON)
option(ENABLE_SSE4_2 "Enables SSE4_2 optimizations on x86/x86_64 targets." ON)
option(ENABLE_AVX "Enables AVX optimizations on x86/x86_64 targets." ON)
option(ENABLE_AVX2 "Enables AVX2 optimizations on x86/x86_64 targets." ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(main main.cpp linalg.cpp thread_pool.cpp)
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})

add_executable(strided_array_test strided_array_test.cpp)
target_link_libraries(strided_array_test ${CMAKE_THREAD_LIBS_INIT})

add_executable(thread_pool_test thread_pool_test.cpp thread_pool.cpp)
target_link_libraries(thread_pool_test ${CMAKE_THREAD_LIBS_INIT})

