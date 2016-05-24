# CMakeLists.txt
#
# W2MHS in ITK
# 
# Shahrzad Moeiniyan Bagheri + Andrew Janke


cmake_minimum_required(VERSION 2.8)
 
project(w2mhs-itk)
 
find_package(ITK REQUIRED)
find_package(OpenCV REQUIRED)

include(${ITK_USE_FILE})

SET(GCC_COMPILE_FLAGS "-std=gnu++11 -fpermissive")

add_definitions(${GCC_COMPILE_FLAGS})

add_executable(w2mhs-itk w2mhs-itk.cxx)

target_link_libraries(w2mhs-itk #changed to match above
  ${Glue} ${ITK_LIBRARIES} ${OpenCV_LIBS})

