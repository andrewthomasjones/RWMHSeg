# CMakeLists.txt
#
# RWMHSeg
# 
# Andrew Jones
#
# using code from w2mhs-itk by Shahrzad Moeiniyan Bagheri + Andrew Janke 2015

cmake_minimum_required(VERSION 2.8)
 
project(RWMHSeg)
 
find_package(ITK REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Boost 
	COMPONENTS 	program_options
 	REQUIRED)

include_directories(${BOOST_INCLUDEDIR})
link_directories(${BOOST_LIBRARYDIR}) 

include(${ITK_USE_FILE})

SET(GCC_COMPILE_FLAGS "-std=gnu++11 -fpermissive")
add_definitions(${GCC_COMPILE_FLAGS})

add_executable(RWMHSeg RWMHSeg.cxx)

target_link_libraries(RWMHSeg #changed to match above
  	${Glue} 
	${ITK_LIBRARIES} 
	${OpenCV_LIBS} 
	${Boost_LIBRARIES}
)

