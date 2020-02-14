include_directories(${Boost_INCLUDE_DIRS})
include_directories(${xGEMM_SOURCE_DIR}/inc)

add_executable(main main.cpp linalg.cpp thread_pool.cpp)
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})

add_executable(strided_array_test strided_array_test.cpp)
target_link_libraries(strided_array_test ${CMAKE_THREAD_LIBS_INIT})

add_executable(thread_pool_test thread_pool_test.cpp thread_pool.cpp)
target_link_libraries(thread_pool_test ${CMAKE_THREAD_LIBS_INIT})

