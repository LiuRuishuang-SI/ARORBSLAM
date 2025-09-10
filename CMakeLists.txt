cmake_minimum_required(VERSION 3.10)
project(ARORBSLAM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(HAVE_PANGOLIN_TARGET FALSE)
find_package(Pangolin QUIET)
if (Pangolin_FOUND)
  if (TARGET Pangolin::pangolin)
    set(HAVE_PANGOLIN_TARGET TRUE)
  endif()
endif()
if (NOT HAVE_PANGOLIN_TARGET)
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(PANGOLIN QUIET pangolin)
    if (PANGOLIN_FOUND)
      set(PANGOLIN_INCLUDE_DIRS ${PANGOLIN_INCLUDE_DIRS} ${PANGOLIN_INCLUDEDIR})
      set(PANGOLIN_LIBRARIES    ${PANGOLIN_LIBRARIES}    ${PANGOLIN_LIBRARIES})
    endif()
  endif()
endif()
if (NOT HAVE_PANGOLIN_TARGET AND NOT PANGOLIN_FOUND)
  find_path(PANGOLIN_INCLUDE_DIRS pangolin/pangolin.h PATHS /usr/local/include /usr/include)
  find_library(PANGOLIN_LIBRARIES NAMES pangolin PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
  if (PANGOLIN_INCLUDE_DIRS AND PANGOLIN_LIBRARIES)
    set(PANGOLIN_FOUND TRUE)
  endif()
endif()
if (NOT HAVE_PANGOLIN_TARGET AND NOT PANGOLIN_FOUND)
  message(FATAL_ERROR "Pangolin not found")
endif()

find_package(OpenGL REQUIRED)
find_package(GLEW QUIET)
if (NOT GLEW_FOUND)
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(GLEW QUIET glew)
  endif()
endif()
find_package(X11 QUIET)

set(ORB_DIR /workspace/ORB_SLAM3)
set(ORB_INC_DIRS
    ${ORB_DIR}
    ${ORB_DIR}/include
    ${ORB_DIR}/include/CameraModels
    ${ORB_DIR}/Thirdparty/DBoW2/DBoW2
    ${ORB_DIR}/Thirdparty/g2o/g2o
    ${ORB_DIR}/Thirdparty/Sophus
)
set(ORB_LIB_DIR ${ORB_DIR}/lib)
find_library(ORB_SLAM3_LIB NAMES ORB_SLAM3 PATHS ${ORB_LIB_DIR} NO_DEFAULT_PATH)
if(NOT ORB_SLAM3_LIB)
  message(FATAL_ERROR "libORB_SLAM3.so not found under ${ORB_LIB_DIR}")
endif()

add_library(plane_detector src/plane_detector.cpp)
target_include_directories(plane_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(plane_detector PUBLIC Eigen3::Eigen)

add_executable(stereo_euroc_plane src/stereo_euroc_plane.cpp)
target_include_directories(stereo_euroc_plane SYSTEM PRIVATE ${ORB_INC_DIRS})
target_include_directories(stereo_euroc_plane PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (HAVE_PANGOLIN_TARGET)
  target_link_libraries(stereo_euroc_plane PRIVATE
    plane_detector
    ${ORB_SLAM3_LIB}
    Pangolin::pangolin
    ${OpenCV_LIBS}
    OpenGL::GL
  )
else()
  target_include_directories(stereo_euroc_plane SYSTEM PRIVATE ${PANGOLIN_INCLUDE_DIRS})
  target_link_libraries(stereo_euroc_plane PRIVATE
    plane_detector
    ${ORB_SLAM3_LIB}
    ${PANGOLIN_LIBRARIES}
    ${OpenCV_LIBS}
    OpenGL::GL
  )
endif()

if (GLEW_FOUND)
  if (TARGET GLEW::GLEW)
    target_link_libraries(stereo_euroc_plane PRIVATE GLEW::GLEW)
  elseif (DEFINED GLEW_LIBRARIES)
    target_link_libraries(stereo_euroc_plane PRIVATE ${GLEW_LIBRARIES})
    if (DEFINED GLEW_INCLUDE_DIRS)
      target_include_directories(stereo_euroc_plane SYSTEM PRIVATE ${GLEW_INCLUDE_DIRS})
    endif()
  endif()
endif()

if (X11_FOUND)
  target_link_libraries(stereo_euroc_plane PRIVATE X11::X11)
endif()

set_target_properties(stereo_euroc_plane PROPERTIES
  BUILD_RPATH "${ORB_LIB_DIR};/usr/local/lib"
  INSTALL_RPATH "${ORB_LIB_DIR};/usr/local/lib"
)

target_compile_options(stereo_euroc_plane PRIVATE -Wno-deprecated-declarations)
