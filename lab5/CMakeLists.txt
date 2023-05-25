cmake_minimum_required(VERSION 3.16.3)

project(cluster C)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

add_library(task_queue STATIC src/task_queue.c)
target_include_directories(task_queue
        PUBLIC include)

add_executable(cluster src/main.c)
target_include_directories(cluster
        PUBLIC include
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(cluster
        PUBLIC task_queue
        PUBLIC MPI::MPI_C
        PUBLIC Threads::Threads)