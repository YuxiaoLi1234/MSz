cmake_minimum_required(VERSION 3.19)

project(MSz LANGUAGES CXX C)

include(GNUInstallDirs)

option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ENABLE_OPENMP "Enable OpenMP acceleration" OFF)
option(ENABLE_ZSTD "Enable Zstd compression" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

if (ENABLE_ZSTD)
    find_package(zstd REQUIRED)
endif()

if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(USE_CUDA TRUE)

    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if (ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        set(USE_OPENMP TRUE)
    else()
        message(WARNING "OpenMP not found, disabling OpenMP acceleration.")
        set(USE_OPENMP FALSE)
    endif()
endif()

set(SRC_FILES
    ./src/api/MSz.cpp
    ./src/internal/MSz_Global/MSz_globals.cpp
    ./src/internal/MSz_Serial/MSz_serial.cpp
)

if (USE_CUDA)
    list(APPEND SRC_FILES ./src/internal/MSz_CUDA/MSz_CUDA.cu)
endif()

if (USE_OPENMP)
    list(APPEND SRC_FILES ./src/internal/MSz_OMP/MSz_omp.cpp)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(MSz SHARED ${SRC_FILES})

target_compile_features(MSz PRIVATE cxx_std_17)

target_include_directories(MSz PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>       
    $<INSTALL_INTERFACE:include/MSz>                            
)

if (USE_CUDA)
    target_link_libraries(MSz PRIVATE CUDA::cudart)
    target_compile_definitions(MSz PRIVATE CUDA_ENABLED)
    set_target_properties(MSz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if (USE_OPENMP)
    target_link_libraries(MSz PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(MSz PRIVATE OPENMP_ENABLED)
endif()

if (ENABLE_ZSTD)
    target_link_libraries(MSz PRIVATE zstd::libzstd)
endif()




install(TARGETS MSz
        EXPORT MSzTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/api/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/MSz/api
        FILES_MATCHING PATTERN "*.h")


install(EXPORT MSzTargets
        FILE MSzTargets.cmake
        NAMESPACE MSz::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSz)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/MSzConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSz
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSz)


add_executable(MSz_CLI ./src/cli/MSz_CLI.cpp)
target_link_libraries(MSz_CLI PRIVATE MSz pthread)


target_link_libraries(MSz_CLI PRIVATE MSz pthread)
target_compile_features(MSz_CLI PRIVATE cxx_std_17)

install(TARGETS MSz_CLI
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (BUILD_EXAMPLES)
    message(STATUS "Building example programs.")
    add_executable(ex1 examples/ex1_MSz_count_faults.cpp)
    target_link_libraries(ex1 PRIVATE MSz pthread)
    target_compile_features(ex1 PRIVATE cxx_std_17)

    add_executable(ex2 examples/ex2_MSz_derive_apply_edits.cpp)
    target_link_libraries(ex2 PRIVATE MSz pthread)
    target_compile_features(ex2 PRIVATE cxx_std_17)
endif()
