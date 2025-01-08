cmake_minimum_required(VERSION 3.20)

project(MSz LANGUAGES CXX CUDA C)

include(GNUInstallDirs)
find_package(SZ3 REQUIRED)
find_package(ZFP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(ZSTD IMPORTED_TARGET libzstd)

option(ENABLE_CUDA "Enable CUDA acceleration" ON)
if (ENABLE_CUDA)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        set(USE_CUDA TRUE)
    else()
        message(WARNING "CUDA not found, disabling CUDA acceleration.")
        set(USE_CUDA FALSE)
    endif()
endif()

option(ENABLE_OPENMP "Enable OpenMP acceleration" ON)
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

add_library(MSz STATIC ${SRC_FILES})

target_compile_features(MSz PRIVATE cxx_std_17)

target_include_directories(MSz PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(MSz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)


if (USE_CUDA)
    target_link_libraries(MSz PRIVATE CUDA::cudart)
    target_compile_definitions(MSz PRIVATE CUDA_ENABLED)
    set_target_properties(MSz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if (USE_OPENMP)
    target_link_libraries(MSz PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(MSz PRIVATE OPENMP_ENABLED)
endif()

target_link_libraries(MSz PRIVATE PkgConfig::ZSTD zfp::zfp SZ3::SZ3)

install(TARGETS MSz
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
