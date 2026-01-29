cmake_minimum_required(VERSION 3.20)

project(MSz LANGUAGES CXX C VERSION 0.0.2)

include(GNUInstallDirs)

option(MSZ_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(MSZ_ENABLE_OPENMP "Enable OpenMP acceleration" OFF)
option(MSZ_ENABLE_ZSTD "Enable Zstd compression" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_SHARED_LIBS "prefer building shared libraies" ON)

# Define the MSz shared library
add_library(MSz
    ./src/api/MSz.cpp
    ./src/internal/MSz_Global/MSz_globals.cpp
    ./src/internal/MSz_Serial/MSz_serial.cpp
)
configure_file(src/internal/msz_config.h.in ${CMAKE_BINARY_DIR}/include/MSz_config.h)
target_compile_features(MSz PRIVATE cxx_std_17)

if(UNIX AND NOT APPLE)
    target_link_options(MSz PUBLIC "-static-libstdc++" "-static-libgcc")
endif()

target_include_directories(MSz PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>   
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_Global>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_OMP>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_Serial>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/internal/MSz_CUDA>
    $<INSTALL_INTERFACE:include/MSz> 
)

# MSz CLI executable
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(MSz_CLI ./src/cli/MSz_CLI.cpp)
target_link_libraries(MSz_CLI PRIVATE MSz Threads::Threads  )

target_compile_features(MSz_CLI PRIVATE cxx_std_17)

# Examples
if (BUILD_EXAMPLES)
    message(STATUS "Building example programs.")
    add_executable(ex1 examples/ex1_MSz_count_faults.cpp)
    target_link_libraries(ex1 PRIVATE MSz Threads::Threads)
    target_compile_features(ex1 PRIVATE cxx_std_17)

    add_executable(ex2 examples/ex2_MSz_derive_apply_edits.cpp)
    target_link_libraries(ex2 PRIVATE MSz Threads::Threads)
    target_compile_features(ex2 PRIVATE cxx_std_17)
    
    add_executable(ex3 examples/ex3_MSz_zstd_omp.cpp)
    target_link_libraries(ex3 PRIVATE MSz Threads::Threads)
    target_compile_features(ex3 PRIVATE cxx_std_17)
endif()

# CUDA linkage
if (MSZ_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    target_sources(MSz PRIVATE ./src/internal/MSz_CUDA/MSz_CUDA.cu)
    target_compile_definitions(MSz PRIVATE MSZ_ENABLE_CUDA=1)
    target_link_libraries(MSz PRIVATE CUDA::cudart)
    set_target_properties(MSz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(MSz PROPERTIES CUDA_ARCHITECTURES "native")
endif()

# OpenMP linkage
if (MSZ_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(MSz PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(MSz PRIVATE MSZ_ENABLE_OPENMP=1)
    target_sources(MSz PRIVATE ./src/internal/MSz_OMP/MSz_omp.cpp)
endif()

if (MSZ_ENABLE_ZSTD)
    find_package(zstd REQUIRED)
    if(TARGET zstd::libzstd)
        target_link_libraries(MSz PRIVATE zstd::libzstd)
    elseif(TARGET zstd::libzstd_shared)
        target_link_libraries(MSz PRIVATE zstd::libzstd_shared)
    elseif(TARGET zstd::libzstd_static)
        target_link_libraries(MSz PRIVATE zstd::libzstd_static)
    else()
        target_link_libraries(MSz PRIVATE zstd)
    endif()
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
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/MSz_config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/MSz/
        )

install(EXPORT MSzTargets
        FILE MSzTargets.cmake
        NAMESPACE MSz::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSz)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfigVersion.cmake"
    VERSION 0.2
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


install(TARGETS MSz_CLI
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Python bindings
option(BUILD_PYTHON "Build Python bindings" ON)
if (BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(msz src/python/msz_py.cpp)
    target_link_libraries(msz PRIVATE MSz)
    
    # Statically link libstdc++ and libgcc to avoid version conflicts in environments like Conda
    if(UNIX AND NOT APPLE)
        target_link_options(msz PRIVATE "-static-libstdc++" "-static-libgcc")
    endif()

    # Set output directory for the python module to the project root or build dir
    set_target_properties(msz PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    # Installation rules for Python module
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        install(TARGETS msz DESTINATION ${Python3_SITELIB})
        message(STATUS "Python module msz will be installed to: ${Python3_SITELIB}")
    else()
        install(TARGETS msz DESTINATION ${CMAKE_INSTALL_LIBDIR}/python)
    endif()
endif()