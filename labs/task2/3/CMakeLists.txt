cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 🔹 Опция для выбора варианта сборки
set(SUM "0" CACHE STRING "Build variant: 0=all, 1=var1, 2=var2, 3=var3")
find_package(OpenMP REQUIRED)

# 🔹 Вывод текущей конфигурации
message(STATUS "Building variant: ${SUM}")

# 🔹 Сборка в зависимости от выбора
if(SUM STREQUAL "0")
    # Все три варианта
    add_executable(variant1 var1_for_omp.cpp)
    target_link_libraries(variant1 PRIVATE OpenMP::OpenMP_CXX)

    add_executable(variant2 var2_parallel_omp.cpp)
    target_link_libraries(variant2 PRIVATE OpenMP::OpenMP_CXX)

    add_executable(variant3 var3_shedule_omp.cpp)
    target_link_libraries(variant3 PRIVATE OpenMP::OpenMP_CXX)

elseif(SUM STREQUAL "1")
    add_executable(variant1 var1_for_omp.cpp)
    target_link_libraries(variant1 PRIVATE OpenMP::OpenMP_CXX)

elseif(SUM STREQUAL "2")
    add_executable(variant2 var2_parallel_omp.cpp)
    target_link_libraries(variant2 PRIVATE OpenMP::OpenMP_CXX)

elseif(SUM STREQUAL "3")
    add_executable(variant3 var3_shedule_omp.cpp)
    target_link_libraries(variant3 PRIVATE OpenMP::OpenMP_CXX)

else()
    message(FATAL_ERROR "Invalid SUM value: ${SUM}. Use 0, 1, 2, or 3")
endif()