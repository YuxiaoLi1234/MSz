cmake_minimum_required(VERSION 3.20)

project(MSz LANGUAGES CXX CUDA C)

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
find_package(SZ3 REQUIRED)
find_package(ZFP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(ZSTD IMPORTED_TARGET libzstd)


add_library(MSz STATIC
    ./include/api/MSz.cpp
    ./include/internal/MSz_CUDA/MSz.cu
    ./include/internal/MSz_Global/MSz_globals.cpp
    ./include/internal/MSz_Serial/MSz_serial.cpp
    ./include/internal/internal.cpp
    ./include/internal/MSz_OMP/MSz_omp.cpp
)


target_compile_features(MSz PRIVATE cxx_std_17)
target_compile_options(MSz PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler -Wno-unused-variable>)
target_link_libraries(MSz PRIVATE PkgConfig::ZSTD zfp::zfp SZ3::SZ3 CUDA::cudart OpenMP::OpenMP_CXX)
set_target_properties(MSz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


install(TARGETS MSz
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  
)


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

