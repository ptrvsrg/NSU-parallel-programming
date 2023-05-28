cmake_minimum_required(VERSION 3.16.3)

project(openmp_slae C)

find_package(OpenMP REQUIRED)

# Add version 1
add_executable(openmp_slae_1 src/openmp_slae_1.c)
target_include_directories(openmp_slae_1
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(openmp_slae_1
        PUBLIC OpenMP::OpenMP_C
        PUBLIC m)

# Add version 2
add_executable(openmp_slae_2 src/openmp_slae_2.c)
target_include_directories(openmp_slae_2
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(openmp_slae_2
        PUBLIC OpenMP::OpenMP_C
        PUBLIC m)

# Add version with custom schedule
add_executable(openmp_slae_schedule src/openmp_slae_schedule.c)
target_include_directories(openmp_slae_schedule
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(openmp_slae_schedule
        PUBLIC OpenMP::OpenMP_C
        PUBLIC m)

# Tests
enable_testing()
set(OMP_NUM_THREADS 1 2 4 8 16)
set(SCHEDULE_TYPE static dynamic guided)
set(SCHEDULE_CHUNK_SIZE 0 100 200 400 800)

# Tests for version 1
foreach (OMP_NUM_THREADS ${OMP_NUM_THREADS})
    add_test(
            NAME test_version_1_omp_num_threads_${OMP_NUM_THREADS}
            COMMAND ./openmp_slae_1)
    set_property(
            TEST test_version_1_omp_num_threads_${OMP_NUM_THREADS}
            PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${OMP_NUM_THREADS};OMP_SCHEDULE=static")
endforeach (OMP_NUM_THREADS)

# Tests for version 2
foreach (OMP_NUM_THREADS ${OMP_NUM_THREADS})
    add_test(
            NAME test_version_2_omp_num_threads_${OMP_NUM_THREADS}
            COMMAND ./openmp_slae_2)
    set_property(
            TEST test_version_2_omp_num_threads_${OMP_NUM_THREADS}
            PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${OMP_NUM_THREADS};OMP_SCHEDULE=static")
endforeach (OMP_NUM_THREADS)

# Tests for version with custom schedule
foreach (SCHEDULE_TYPE ${SCHEDULE_TYPE})
    foreach (SCHEDULE_CHUNK_SIZE ${SCHEDULE_CHUNK_SIZE})
        add_test(
                NAME test_version_with_custom_schedule_${SCHEDULE_TYPE}_${SCHEDULE_CHUNK_SIZE}
                COMMAND ./openmp_slae_schedule)
        set_property(
                TEST test_version_with_custom_schedule_${SCHEDULE_TYPE}_${SCHEDULE_CHUNK_SIZE}
                PROPERTY ENVIRONMENT "OMP_NUM_THREADS=16;OMP_SCHEDULE=${SCHEDULE_TYPE},${SCHEDULE_CHUNK_SIZE}")
    endforeach (SCHEDULE_CHUNK_SIZE)
endforeach (SCHEDULE_TYPE)