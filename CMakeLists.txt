cmake_minimum_required(VERSION 3.20)

project(MSz LANGUAGES CXX C)

include(GNUInstallDirs)
# find_package(PkgConfig REQUIRED)
# pkg_search_module(ZSTD IMPORTED_TARGET libzstd)

find_package (zstd REQUIRED)

option(ENABLE_CUDA "Enable CUDA acceleration" OFF)

if (ENABLE_CUDA)
    enable_language (CUDA)
    set (USE_CUDA TRUE)
    # find_package(CUDAToolkit)
    # if (CUDAToolkit_FOUND)
    #     set(USE_CUDA TRUE)
    #     message(STATUS "CUDA support enabled in MSz.")
    # else()
    #     message(WARNING "CUDA not found, disabling CUDA acceleration.")
    #     set(USE_CUDA FALSE)
    # endif() # 结束 CUDAToolkit_FOUND 检查
endif() # 结束 ENABLE_CUDA 检查

option(ENABLE_OPENMP "Enable OpenMP acceleration" OFF)
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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> 
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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

# target_link_libraries(MSz PRIVATE PkgConfig::ZSTD)
target_link_libraries(MSz PRIVATE zstd::libzstd)

install(TARGETS MSz
        EXPORT MSzTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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
    "${CMAKE_CURRENT_SOURCE_DIR}/MSzConfig.cmake.in" # 位于根目录
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSz
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MSzConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MSz)
