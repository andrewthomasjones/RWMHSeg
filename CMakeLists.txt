# CMakeLists.txt
#
# W2MHS in ITK
# 
# Shahrzad Moeiniyan Bagheri + Andrew Janke

#Andrew Jones added Boost,  June 2016

cmake_minimum_required(VERSION 2.8)
 
project(w2mhs-itk)
 
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

add_executable(w2mhs-itk w2mhs-itk.cxx)

target_link_libraries(w2mhs-itk #changed to match above
  	${Glue} 
	${ITK_LIBRARIES} 
	${OpenCV_LIBS} 
	${Boost_LIBRARIES}
)

