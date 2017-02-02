# RWMHSeg - Robust White Matter Hyperintensity Segmentaion
An implementation of a new robust method for White Matter Hyperintensity Segmentaion, using the pipeline of the W2MHS white matter hyperintensity segmentation method, as reproduced using ITK in the W2MHS-ITK project.

Install/compile instructions for Ubuntu:

# Install opencv
    $ apt-get install libopencv-dev

# Install Glue
    $ apt-get install glue-sprite

# Install Boost	 
    $ apt-get install libboost-all-dev

# Install ITK	
Instructions can be found [here](https://itk.org/Wiki/ITK_Configuring_and_Building_for_Ubuntu_Linux#Installing_ITK_from_Ubuntu_packages).

#Build:
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
