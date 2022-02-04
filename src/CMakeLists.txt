cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)

project(EsnMain)

set( CMAKE_CXX_FLAGS "-Ofast" )
set( CMAKE_CXX_FLAGS "-DARMA_NO_DEBUG" )


if(DEFINED ENV{ARMADILLO_DIR})
   message("-- Found armadillo directory")
else()
   message(FATAL_ERROR "ERROR: Set ARMADILLO_DIR")
endif()

if(DEFINED ENV{CXXOPTS_DIR})
   message("-- Found cxxopts directory")
else()
   message(FATAL_ERROR "ERROR: Set CXXOPTS_DIR")
endif()


add_executable( EsnMain
                EsnMain.cxx EsnOpts.cxx Esn.cxx)

target_include_directories( EsnMain
                            PUBLIC $ENV{CXXOPTS_DIR}/include/
                            PUBLIC $ENV{ARMADILLO_DIR}/include/)

target_link_directories( EsnMain 
                         PUBLIC $ENV{ARMADILLO_DIR}/lib )

target_link_libraries( EsnMain 
                       armadillo )

