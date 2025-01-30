cmake_minimum_required(VERSION 3.19)

project(MSz LANGUAGES CXX C)

include(GNUInstallDirs)

option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(ENABLE_OPENMP "Enable OpenMP acceleration" OFF)
option(ENABLE_ZSTD "Enable Zstd compression" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# Zstd integration using pkg-config
if (ENABLE_ZSTD)
    find_package(zstd REQUIRED)
endif()

# CUDA integration
if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(USE_CUDA TRUE)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_COMPATIBLE_COMPUTE_CAPABILITIES})
    else()
        message(FATAL_ERROR "CUDA architectures must be manually specified for CUDA versions < 11")
    endif()
endif()

# OpenMP integration
if (ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        set(USE_OPENMP TRUE)
    else()
        message(WARNING "OpenMP not found, disabling OpenMP acceleration.")
        set(USE_OPENMP FALSE)
    endif()
endif()

# Source files
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

# Define the MSz shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(MSz SHARED ${SRC_FILES})

target_compile_features(MSz PRIVATE cxx_std_17)

target_include_directories(MSz PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   # 项目中的公共头文件目录
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_Global>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_OMP>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_Serial>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_CUDA>
    $<INSTALL_INTERFACE:include/MSz> 
)

# CUDA linkage
if (USE_CUDA)
    target_link_libraries(MSz PRIVATE CUDA::cudart)
    target_compile_definitions(MSz PRIVATE CUDA_ENABLED)
    set_target_properties(MSz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# OpenMP linkage
if (USE_OPENMP)
    target_link_libraries(MSz PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(MSz PRIVATE OPENMP_ENABLED)
endif()

if (ENABLE_ZSTD)
    target_include_directories(MSz PRIVATE ${Zstd_INCLUDE_DIRS})
    target_link_directories(MSz PRIVATE ${Zstd_LIBRARY_DIRS})
    target_link_libraries(MSz PRIVATE ${Zstd_LIBRARIES})
    target_compile_options(MSz PRIVATE ${Zstd_CFLAGS_OTHER})
endif()

# Installation configuration
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

# MSz CLI executable
add_executable(MSz_CLI ./src/cli/MSz_CLI.cpp)
target_link_libraries(MSz_CLI PRIVATE MSz pthread)

target_compile_features(MSz_CLI PRIVATE cxx_std_17)

install(TARGETS MSz_CLI
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Examples
if (BUILD_EXAMPLES)
    message(STATUS "Building example programs.")
    add_executable(ex1 examples/ex1_MSz_count_faults.cpp)
    target_link_libraries(ex1 PRIVATE MSz pthread)
    target_compile_features(ex1 PRIVATE cxx_std_17)

    add_executable(ex2 examples/ex2_MSz_derive_apply_edits.cpp)
    target_link_libraries(ex2 PRIVATE MSz pthread)
    target_compile_features(ex2 PRIVATE cxx_std_17)
endif()
