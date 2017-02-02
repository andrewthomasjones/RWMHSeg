# RWMHSeg - Robust White Matter Hyperintensity Segmentaion
An implementation of a new robust method for White Matter Hyperintensity Segmentaion, using the pipeline of the W2MHS white matter hyperintensity segmentation method, as reproduced using ITK in the W2MHS-ITK project.

Install/compile instructions for Ubuntu:

# Install opencv
    $ apt-get install libopencv-dev

# Install Glue
    $ apt-get install glue-sprite

# Install Boost	 
    $ apt-get install libboost-all-dev
    
# Install cmake    
    $ sudo apt-get install cmake
    
# Install VTK    
    $ sudo apt-get install libvtk5-dev 

# Install ITK	
Instructions can be found [here](https://itk.org/Wiki/ITK_Configuring_and_Building_for_Ubuntu_Linux#Building_and_installing_ITK_from_source_code.)
Note that you will need to install ITK from source, as some of the functions used are not available through the ubuntu aptitude install. 

#Build:
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
