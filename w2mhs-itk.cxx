/* W2MHS in ITK
 *
 *  re-implementation of W2MHS + improvements in ITK
 * 
 *  Shahrzad Moeiniyan Bagheri - shahrzad.moeiniyanbagheri@uqconnect.edu.au
 *  Andrew Janke - a.janke@gmail.com
 *  Center for Advanced Imaging
 *  The University of Queensland
 *
 *  Copyright (C) 2015 Shahrzad Moeiniyan Bagheri and Andrew Janke
 *  This package is licenced under the AFL-3.0 as per the original 
 *  W2MHS MATLAB implmentation.
 */


/*
 * 
 * additional edits by andrew jones
 * andrewthomasjones@gmail.com
 * 2016
 * main changes:
 * input arg structure changed
 * new modes for testig and training
 * new models
 * added boost
 * refactored to use strings over *char mostly
 */
 
//ITK HEADERS
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkNiftiImageIO.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhood.h"
#include "itkConvolutionImageFilter.h"
#include "itkConstantBoundaryCondition.h"
#include "itkSubtractImageFilter.h"
#include "itkNeighborhoodOperator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageToHistogramFilter.h"
#include "itkHistogramThresholdImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkBinaryContourImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkPowImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkNeighborhoodOperator.h"
#include "itkMRFImageFilter.h"
#include "itkMinimumDecisionRule.h"
#include "itkDistanceToCentroidMembershipFunction.h"
#include "itkManhattanDistanceMetric.h"
#include "itkEuclideanDistanceMetric.h"
//MISC
#include "array"
#include "iostream"
#include "fstream"
#include "cstdio"
#include "cstdlib"
#include "cv.h"       // opencv general include file
#include "ml.h"
#include "string.h"
#include "vector"
#include "unistd.h"
#include "getopt.h"
#include "vector"


//BOOST
#include "boost/math/distributions/students_t.hpp"
#include "boost/program_options.hpp"
#include "boost/math/tools/roots.hpp"
#include "boost/math/special_functions/digamma.hpp"

//NAMESPACES
using namespace cv;
using namespace boost::program_options;
using namespace boost::math;

//IMAGE TYPES
typedef itk::Image<float,3> ImageType;
typedef ImageType::Pointer ImagePointer;
typedef itk::Image<float,2> ImageType2D;
typedef ImageType2D::Pointer ImagePointer2D;
typedef std::pair <std::vector<double>, std::vector<double> > vecPair; 

//FILTER, ETC TYPES
typedef itk::MaskImageFilter< ImageType, ImageType > MaskFilterType;
typedef itk::ConstantBoundaryCondition<ImageType>  BoundaryConditionType;
typedef itk::ConstNeighborhoodIterator< ImageType , BoundaryConditionType> NeighborhoodIteratorType;
typedef itk::Neighborhood< ImageType > NeighborhoodType;
typedef itk::ConvolutionImageFilter<ImageType> ConvolutionType;

//declare the methods' signature here
ImagePointer NiftiReader(char * inputFile);
ImagePointer NiftiReader(std::string inputFile);
bool NiftiWriter(ImagePointer input,std::string outputFile);
ImagePointer GetPatch(ImagePointer input,ImageType::IndexType centreIdx,unsigned int ROIsize);
void MarginateImage(ImagePointer input,int marginWidth);

////general methods
void ShowCommandLineHelp();

////image operators
ImagePointer DivideImageByConstant(ImagePointer input,double constant);
ImagePointer MultiplyTwoImages(ImagePointer input1,ImagePointer input2);
ImagePointer AddImageToImage(ImagePointer input1,ImagePointer input2,bool isSubraction);
float SumImageVoxels(ImagePointer input);
ImagePointer PowerImageToConst(ImagePointer input, float constPower);
ImagePointer MinMaxNormalisation(ImagePointer input);
ImagePointer InvertImage(ImagePointer input, int maximum);
double GetHistogramMax(ImagePointer inputImage,unsigned int binsCount);
ImagePointer ThresholdImage(ImagePointer input, float lowerThreshold, float upperThreshold);
ImagePointer2D BinarizeThresholdedImage(ImagePointer2D input2D, float lowerThreshold, float upperThreshold);
ImagePointer2D Get2DBinaryObjectsBoundaries(ImagePointer2D input);
ImagePointer2D Get2DSlice(ImagePointer input3D,int plane, int slice);
std::vector<ImageType2D::IndexType> FindIndicesByIntensity(ImagePointer2D input,float intensity);
ImagePointer2D LabelBinaryObjects(ImagePointer2D input2D, int* objectCount);
double GetMinIndexDistanceFromObjects(ImageType2D::IndexType inputIdx,ImagePointer2D objectsBoundaries);

////feature extraction
ImagePointer CreateGaussianKernel(float variance,int width);
ImagePointer CreateLaplacianOfGKernel(float variance,int width);
std::list<ImagePointer> CreateDifferenceOfGLKernels(bool GaussianOrLaplacian,std::array<int,2> width,std::array<double,4> baseVarianceArray);
ImagePointer CreateSobelKernel();
ImagePointer ConvolveImage(ImagePointer input,ImagePointer kernel,bool normaliseOutput);
ImagePointer ConvolveImage2(ImagePointer patch,ImagePointer kernel);
std::list<ImagePointer> GetAllKernels();
void AppendToPatchFeatureVector(ImagePointer patch,Mat patchFeatureMat, int startIdx);
void CreatePatchFeatureVector(ImagePointer patch, Mat patchFeatureMat);

////Random Forest functions
bool LoadTrainingDataset(std::string  trainingFilename, Mat trainingFeatures, Mat trainingLabels,int samplesCount,int featuresCount);
bool LoadTestingDataset(std::string  testingFilename, Mat testingFeatures, Mat testingLabels,int samplesCount,int featuresCount);
CvRTrees* GenOrLoadRFModel(std::string  modelFilename, Mat trainingFeatures, Mat trainingLabels);
CvRTrees* GenOrLoadRFModel(std::string  modelFilename);

ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount, std::string rfSegOutFilename);
ImagePointer ClassifyWMHs(ImagePointer WMModStripImg, std::string rfSegOutFilename);
ImagePointer CreateWMHPmap(ImagePointer rfSegmentedImg, ImagePointer refImg, std::string gcFilename, std::string pmapFilename);
void QuantifyWMHs(float pmapCut, ImagePointer pmapImg, std::string ventricleFilename, std::string outputFilename);


////FOR TEST
void CreatePatchFeatureVector(ImagePointer patch, Mat patchFeatureMat,std::string outputFilename,float classLabel);   //FOR TESTING AND DEBUGGING
void CalculateMSE(ImagePointer img1,ImagePointer img2);


//New functions - Andy 2016
Mat getFeatureVector(ImagePointer WMModStripImg, ImagePointer BRAVOImg, int featuresCount);
Mat getLocationVector(ImagePointer WMModStripImg);

void getFeatureOut(std::string WMStrippedFilename, std::string BRAVOFilename,std::string WMMaskFilename, std::string quantResultFilename2, int numberOfFeatures);
void ReadSubFolders(std::string folderName, std::string foldersList);
void CreateTrainingDataset(std::string WMFilename,std::string pmapFilename,std::string segoutFilename,std::string featuresFilename);
std::string renamer(std::string baseName, std::string suffix);

vecPair fitTdist(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter, double a, double b);
vecPair fitMest(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter);


//math functions
double phi(double d);
template <typename T> int sgn(T val);
double Digamma(double x);
double df_eq_est (double v, double v_sum);

//function class with operator for root finder for df of t-dist
class df_eq_func{
	public:
		df_eq_func(double v) :  v_sum(v){};
	
		double getSumV();
		
		double operator()(double v) {
			
			double temp = -digamma(0.5*v) + std::log(0.5*v) + v_sum + digamma((v+1.)/2.) - std::log((v+1.)/2.)+1.;
			//std::cout << "v: " << v << " v_sum: " << v_sum << " temp: " << temp << std::endl;
			return temp;
			//return df_eq_est(v,v_sum);
			
		}
		
	private:
		double v_sum;
};

double df_eq_func::getSumV()
{
    return v_sum;
}


//class MembershipFunctionThreshold





std::string vecToString(vector<string> v);


//new models
ImagePointer ClassifyWMHsT(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour,  double p_thresh_const, double min_df, double max_df);
ImagePointer ClassifyWMHsM(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double out_thresh_const, int max_iter, double tol);
ImagePointer ClassifyWMHsMRF(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double out_thresh_const, int max_iter, double tol);

////////////////////////////////////////////////   //////////////////////////////////////////////////////////

		
int main(int argc, char *argv[])
{		//falgs
		bool testFlag = false;
		bool justGetFeat = false;
		bool createNewTrainingSet = false;
		bool createNewRFModel = false;
		
		bool classicFlag = false;
		bool TFlag = false;
		bool MFlag = false;
		bool twoDflag =  false; 
		  	   		
		int numberOfTrainingSamples;
		int min_neighbours ;
		int numberOfFeatures = 2000;
		int mode ;
		
		double pmapCut;
		double p_thresh_const;
		double d_thresh_const;
		
		double a;
	    double b;
			
		//names of various files, input and output
		std::string WMStrippedFilename;
		std::string GMCSFStippedFilename;
		std::string ventricleBinFilename; 
		std::string rfmodelFilename;
		
		//output
		std::string quantResultFilename;
		std::string pmapOutFilename;
		std::string segOutFilename;
		
		
		//other
		std::string folderName;
		std::string subFolderName;
		std::string trainingFilename;
		std::string quantResultFilename2;
		std::string WMMaskFilename;
		std::string BRAVOFilename;
		
		try
		{
			options_description desc{"Options"};
			desc.add_options()
				("help,h", "Help screen")
				
				("min_n,n", value<int>(), "Minimum neighbours required for WMH pixels in new model ")
				("p_cut,c", value<double>(), "pmap cut-off for EV calculations in RF model")
				("p_thresh,x", value<double>(), "Significance cut off for t-dist based model. Lower values mean less pixels classified as WMH ")
				("d_thresh,d", value<double>(), "Distance cut off for robust normal (M-Estimator) model. Higher values mean less pixels classified as WMH")
				
				("min_v,a", value<double>()->default_value(0.01), "min df fotr t-dist")
				("max_v,b", value<double>()->default_value(10.), "max df fotr t-dist")
				
				("wm_strip,w", value< vector<string>> () , "WM Stripped  filename")
				("gm_strip,g", value< vector<string> >() , "GMCSF  filename")
				("vent_mask,v", value< vector<string> >() , "ventricle binary mask filename")
				("rf_mod,m", value< vector<string> >() , "RF Model filename")
				
				("quant,q", value< vector<string> >() , "Quant result filename")
				("pmap,p", value< vector<string> >() , "pmap output  filename")
				("seg,s", value< vector<string> >() , "Segmentation ouput filename")
				
				("z", value<int>()->default_value(1), "test mode --z 0 ")
				
				("train,t", value< vector<string> >() , "training filename")
				("sample", value<int>(), "Number of training samples")
								
				("out,o", value< vector<string>> () , "Feature save filename")
				("bravo,y", value< vector<string>> () , "Filename for 2nd channel image")
				("wm_mask,f", value< vector<string>> () , "Filename for WM mask to extract correct region on 2nd channel image");
	
			variables_map vm;
			store(parse_command_line(argc, argv, desc), vm);
			notify(vm);
			
			if (vm.count("help")) {
				ShowCommandLineHelp();
				return 1;
			}
			
			//mode flags
			if (vm.count("z")){
				
				mode = vm["z"].as<int>();
								
				if(mode==0){
					testFlag= true;
				}
				
								
				
			}
			
			
			
			
			// if use pmap cut value - normal mode
			if (vm.count("p_cut")){
				try{
					pmapCut = vm["p_cut"].as<double>(); 
					classicFlag= true;
					//names of various files, input and output
					WMStrippedFilename = vecToString(vm["wm_strip"].as< vector<string> >());  
					GMCSFStippedFilename = vecToString(vm["gm_strip"].as< vector<string> >()); 
					ventricleBinFilename = vecToString(vm["vent_mask"].as< vector<string> >()); 
					rfmodelFilename = vecToString(vm["rf_mod"].as< vector<string> >()); 
					//output
					quantResultFilename = vecToString(vm["quant"].as< vector<string> >()); 
					pmapOutFilename = vecToString(vm["pmap"].as< vector<string> >()); 
					segOutFilename = vecToString(vm["seg"].as< vector<string> >());
				}catch(...){
					std::cerr << "Unable to read all inputs for classic(RF model) mode. Run --help" <<std::endl;
					return 0;
				} 
				
			}
			
			if (vm.count("p_thresh")){
				try{
					// t-dist model
					TFlag= true;
					//names of various files, input and output
					WMStrippedFilename = vecToString(vm["wm_strip"].as< vector<string> >());  
					GMCSFStippedFilename = vecToString(vm["gm_strip"].as< vector<string> >()); 
					ventricleBinFilename = vecToString(vm["vent_mask"].as< vector<string> >()); 
					min_neighbours = vm["min_n"].as<int>();  //inputs for models
					p_thresh_const = vm["p_thresh"].as<double>();
					a = vm["min_v"].as<double>();
					b = vm["max_v"].as<double>();
					//std::cout<< p_thresh_const <<  vm["p_thresh"].as<double>() << std::endl;
					//output
					quantResultFilename = vecToString(vm["quant"].as< vector<string> >()); 
					segOutFilename = vecToString(vm["seg"].as< vector<string> >()); 
					
					//two d
					if (vm.count("bravo")){
						BRAVOFilename = vecToString(vm["bravo"].as< vector<string> >()); 
						twoDflag = true;
					}
						
						
				}catch(...){
					std::cerr << "Unable to read all inputs for t-distribution mode. Run --help" <<std::endl;
					return 0;
				} 
				
			}
			
			if (vm.count("d_thresh")){ //m-estimator model
				try{
					MFlag= true;
					//names of various files, input and output
					WMStrippedFilename = vecToString(vm["wm_strip"].as< vector<string> >());  
					GMCSFStippedFilename = vecToString(vm["gm_strip"].as< vector<string> >()); 
					ventricleBinFilename = vecToString(vm["vent_mask"].as< vector<string> >()); 
					min_neighbours = vm["min_n"].as<int>();  //inputs for models
					d_thresh_const = vm["d_thresh"].as<double>();
					
					//output
					quantResultFilename = vecToString(vm["quant"].as< vector<string> >()); 
					segOutFilename = vecToString(vm["seg"].as< vector<string> >()); 
					
					//two d
					if (vm.count("bravo")){
						BRAVOFilename = vecToString(vm["bravo"].as< vector<string> >()); 
						twoDflag = true;
					}
				}catch(...){
					std::cerr << "Unable to read all inputs for m-estimator mode. Run --help" <<std::endl;
					return 0;
				} 
				
			}
						
			if (vm.count("sample")){ //new rf mod
				try{
					createNewRFModel = true;
					trainingFilename = vecToString(vm["train"].as< vector<string> >());
					rfmodelFilename = vecToString(vm["rf_mod"].as< vector<string> >());
					numberOfTrainingSamples= vm["sample"].as<int>();  //the training data set size
				}catch(...){
					std::cerr << "Unable to read all inputs for new RF model mode. Run --help" <<std::endl;
					return 0;
				} 
					
				
			}
			
			if (vm.count("out")){ //feature output mode
				try{
					
					justGetFeat = true;
					WMStrippedFilename = vecToString(vm["wm_strip"].as< vector<string> >()); 
					quantResultFilename2 = vecToString(vm["out"].as< vector<string> >()); 
					BRAVOFilename = vecToString(vm["bravo"].as< vector<string> >()); 
					WMMaskFilename = vecToString(vm["wm_mask"].as< vector<string> >());
				}catch(...){
					std::cerr << "Unable to read all inputs for raw feature output mode. Run --help" <<std::endl;
					return 0;
				} 
					
				
			}else{
				try{
					if (vm.count("bravo")){
						BRAVOFilename = vecToString(vm["bravo"].as< vector<string> >()); 
						twoDflag  = true ;
					}
				}catch(...){
					std::cerr << "Second image (T1/Bravo) argument was not correctly read." <<std::endl;
					return 0;
				}
			}
			
			std::cout << std::endl;
			std::cout << "Inputs read successfully." <<std::endl;
	
		}
		catch(...)
		{//fix this up later with more specific errors, currently only needs the correct args to be present for each mode.
			//doesnt check output locations have write access, if files are legit, if numbers are in range, etc etc
			std::cerr << "Unknown failure reading in cmd line args" << std::endl;
			return 0;
		}
	
		
		


		
		
		
		////////////////////////////////////////   testing modes   /////////////////////////////////////////
		
		
		// does some testing  
		if(testFlag)
		{
			
			ImagePointer RFSegOutImage,RFSegOutImage2;  
			ImagePointer inNifti = NiftiReader(WMStrippedFilename); 
					
			std::cout<< "p_thresh_const :  " << p_thresh_const  << std::endl;
			std::cout<< "d_thresh_const :  " << d_thresh_const  << std::endl;
			std::cout<< "min_neighbour :  " << min_neighbours  << std::endl;
    			
			RFSegOutImage=ClassifyWMHsT(inNifti, segOutFilename,  min_neighbours,  p_thresh_const, a, b);
			RFSegOutImage2=ClassifyWMHsM(inNifti, segOutFilename,  min_neighbours,  d_thresh_const, 100, 0.01);
			
			QuantifyWMHs(0.0, RFSegOutImage, ventricleBinFilename, quantResultFilename);
			QuantifyWMHs(0.0, RFSegOutImage2, ventricleBinFilename, quantResultFilename);
			
			std::cout << " done" <<std::endl;
		
			return 0;
		}
		
		//returns features
		if(justGetFeat)
		{  	
			std::cout << "processing... "  << std::endl;
			getFeatureOut(WMStrippedFilename, BRAVOFilename, WMMaskFilename, quantResultFilename2, numberOfFeatures );
			std::cout << "features saved in: " << quantResultFilename2 << std::endl;
			return 0;
		}
		
		//makes new pseudotraining set
		//not currently implemented
		if(createNewTrainingSet)
		{
			std::cout << folderName << " " <<subFolderName <<std::endl;
			ReadSubFolders(folderName,subFolderName);
			std::cout << "Done!" << std::endl;
			return 0;
		}
		
		
		if(createNewRFModel) // fits new RF model
		{
				
				CvRTrees* rfModel=new CvRTrees();   //Random Forest classifier. This model will be saved into a .xml file for later use.
   
				//CV_32FC2 means a 2-channel (complex) floating-point array
				Mat trainingSamples = Mat(numberOfTrainingSamples, numberOfFeatures, CV_32FC1);   //this is to keep the training data
				Mat trainingLabels = Mat(numberOfTrainingSamples, 1, CV_32FC1);               //this is to keep the labels provided in the training data

				std::cout << "numberOfFeatures " << numberOfFeatures << std::endl;
				std::cout << "numberOfTrainingSamples " << numberOfTrainingSamples << std::endl;
				std::cout << "Loading training file..." << std::endl;
			   
				// loading training and testing data setsCreatePatchFeatureVector
				if (LoadTrainingDataset(trainingFilename, trainingSamples, trainingLabels, numberOfTrainingSamples, numberOfFeatures)){
				  
					std::cout << "Creating new model from training file..." << std::endl;
					rfModel=GenOrLoadRFModel(rfmodelFilename,trainingSamples,trainingLabels);
					return 0;
				 
				}else
				{	
					std::cerr << "Training data set couldn't be loaded!" << std::endl;
					return -1;
				}
		}
		
		
		
	  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	   if(classicFlag)
	   {
			CvRTrees* rfModel=new CvRTrees();   //Random Forest classifier. This model will be saved into a .xml file for later use.
		
			rfModel=GenOrLoadRFModel(rfmodelFilename.c_str());
			
			std::cout<< "RF Model"  << std::endl;
			std::cout<< "quant :  " << quantResultFilename  << std::endl;
			std::cout<< "seg :  " << segOutFilename  << std::endl;
			std::cout<< "pmap :  " << pmapOutFilename  << std::endl;
		   		
			ImagePointer inNifti=NiftiReader(WMStrippedFilename);
			//inNifti->SetReleaseDataFlag(true);      //Turn on/off the flags to control whether the bulk data belonging to the outputs of this ProcessObject are released after being used by a downstream ProcessObject. Default value is off. Another options for controlling memory utilization is the ReleaseDataBeforeUpdateFlag.
			
			ImagePointer RFSegOutImage;
			
			RFSegOutImage=ClassifyWMHs(inNifti,rfModel,numberOfFeatures,segOutFilename);   //For TEST USE: RFSegOutImage=ClassifyWMHs(inNifti,rfModel,numberOfFeatures,segOutFilename,testingSamples);
			
			// Creating the probability map image out of the segmented image in the previous step.
			ImagePointer pmap=CreateWMHPmap(RFSegOutImage,inNifti,GMCSFStippedFilename,pmapOutFilename);
		
			QuantifyWMHs(pmapCut,pmap,ventricleBinFilename,quantResultFilename);   //TODO: 'pmapCut' parameter may come as an input command line argument
			std::cout << "DONE!" << std::endl;
			
	}
	
	if(MFlag){
			
			ImagePointer RFSegOutImageM; 
			
			//~ ImagePointer RFSegOutImageM2; 
			//~ ImagePointer RFSegOutImageM3;  
			//~ ImagePointer RFSegOutImageM4; 
			//~ ImagePointer RFSegOutImageM5; 
			
			ImagePointer inNifti = NiftiReader(WMStrippedFilename); 
			
			//for testing sensitivty
			std::string quantResultFilenameM = renamer(quantResultFilename, "_m");
			//~ std::string quantResultFilenameM2 = renamer(quantResultFilename, "_m2");	
			//~ std::string quantResultFilenameM3 = renamer(quantResultFilename, "_m3");	
			//~ std::string quantResultFilenameM4 = renamer(quantResultFilename, "_m4");	
			//~ std::string quantResultFilenameM5 = renamer(quantResultFilename, "_m5");
			
			//~ double  d_thresh_const2 = d_thresh_const + 0.05;
			//~ double  d_thresh_const3 = d_thresh_const + 0.10;
			//~ double  d_thresh_const4 = d_thresh_const + 0.15;
			//~ double  d_thresh_const5 = d_thresh_const + 0.20;
			
				
			//~ std::string segOutFilenameM2 = renamer(segOutFilename, "_m2");	
			//~ std::string segOutFilenameM3= renamer(segOutFilename, "_m3");	
			//~ std::string segOutFilenameM4 = renamer(segOutFilename, "_m4");	
			//~ std::string segOutFilenameM5 = renamer(segOutFilename, "_m5");	
			std::string segOutFilenameM = renamer(segOutFilename, "_m");	
			
			std::cout<< "M-estimator model"  << std::endl;
			std::cout<< "quant :  " << quantResultFilenameM  << std::endl;
			std::cout<< "seg :  " << segOutFilenameM  << std::endl;
											
			std::cout<< "d_thresh_const :  " << d_thresh_const  << std::endl;
			std::cout<< "min_neighbour :  " << min_neighbours  << std::endl;
    			
			
			//~ 
			//~ RFSegOutImageM2=ClassifyWMHsM(inNifti, segOutFilenameM2,  min_neighbours,  d_thresh_const2, 100, 0.0001);
			//~ QuantifyWMHs(0.0, RFSegOutImageM2, ventricleBinFilename, quantResultFilenameM2);
			//~ 
			//~ RFSegOutImageM3=ClassifyWMHsM(inNifti, segOutFilenameM3,  min_neighbours,  d_thresh_const3, 100, 0.0001);
			//~ QuantifyWMHs(0.0, RFSegOutImageM3, ventricleBinFilename, quantResultFilenameM3);
			//~ 
			//~ RFSegOutImageM4=ClassifyWMHsM(inNifti, segOutFilenameM4,  min_neighbours,  d_thresh_const4, 100, 0.0001);
			//~ QuantifyWMHs(0.0, RFSegOutImageM4, ventricleBinFilename, quantResultFilenameM4);
			//~ 
			//~ RFSegOutImageM5=ClassifyWMHsM(inNifti, segOutFilenameM5,  min_neighbours,  d_thresh_const5, 100, 0.0001);
			//~ QuantifyWMHs(0.0, RFSegOutImageM5, ventricleBinFilename, quantResultFilenameM5);
			
			//RFSegOutImage=ClassifyWMHsT(inNifti, segOutFilename,  min_neighbours,  p_thresh_const, a, b);
			RFSegOutImageM=ClassifyWMHsM(inNifti, segOutFilenameM,  min_neighbours,  d_thresh_const, 100, 0.0001);
			QuantifyWMHs(0.0, RFSegOutImageM, ventricleBinFilename, quantResultFilenameM);
			
			
			//QuantifyWMHs(0.0, RFSegOutImage2, ventricleBinFilename, quantResultFilename);
			
			std::cout << "Done" <<std::endl;
		
			
			}
			
	if(TFlag){
			
			ImagePointer RFSegOutImageT; 
			ImagePointer inNifti = NiftiReader(WMStrippedFilename); 
			
			std::string quantResultFilenameT = renamer(quantResultFilename, "_t");	
			std::string segOutFilenameT = renamer(segOutFilename, "_t");
				
			std::cout<< "t-distribution model"  << std::endl;
			
			std::cout<< "quant :  " << quantResultFilenameT  << std::endl;
			std::cout<< "seg :  " << segOutFilenameT  << std::endl;
			
			std::cout<< "p_thresh_const :  " << p_thresh_const  << std::endl;
			std::cout<< "min_neighbour :  " << min_neighbours  << std::endl;
    			
			RFSegOutImageT=ClassifyWMHsT(inNifti, segOutFilenameT,  min_neighbours,  p_thresh_const, a, b);
			//RFSegOutImage2=ClassifyWMHsM(inNifti, segOutFilenameM,  min_neighbours,  d_thresh_const);
			
			
			
			
			
			
	
			
			
			//QuantifyWMHs(0.0, RFSegOutImageM, ventricleBinFilename, quantResultFilenameM);
			QuantifyWMHs(0.0, RFSegOutImageT, ventricleBinFilename, quantResultFilenameT);
			
			std::cout << "Done" <<std::endl;
		
			
			}
	
	return 0;
}//end of main()




void ShowCommandLineHelp()
{
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-h, --help:  Shows command line help." << std::endl;
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << "Required arguements:  " << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-w, --wm_strip: Specifies the .nii file name and path of the WM stripped image."<< std::endl;
	std::cerr << "      " << "-g, --gm_strip: Specifies the .nii file name and path of the GMCSF stripped image." << std::endl;
	std::cerr << "      " << "-v, --vent_mask: Specifies the .nii file name and path of the binary ventricle PVE image." << std::endl<< std::endl;
	
	std::cerr << "      " << "-q, --quant: Specifies the (.txt) file name and path, where WMH quantification results will be saved." << std::endl;
	std::cerr << "      " << "-s, --seg: Specifies the .nii file name and path, where the WMH segmentation results will be saved."<< std::endl;
	std::cerr << std::endl;
	std::cerr << "      " <<  "  *** Must supply at least one of --c, --d, --p as these also function to select the relevant model. ***" << std::endl;
	std::cerr << "      " <<  "  *** Results from new models will append _m and _t to output filenames. ***" << std::endl;
	std::cerr << "=============================================================================="<< std::endl; 
	
	std::cerr << "RF Model Only:  " << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-c, --p_cut: Specifies the minimum probability cut-off for volume calculations using pmap" << std::endl;
	std::cerr << "      " << "-m, --rf_mod: Specifies the .xml file name and path, where the Random Forest model will be saved/loaded."<< std::endl;
	std::cerr << "      " << "-p, --pmap: Specifies the .nii file name and path, where the WMH probability map will be saved." << std::endl;
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	
	std::cerr << "New Models Only:  " << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-n, --min_n: Specifies the minimum neighbours for new methods"<< std::endl;
	std::cerr << "      " << "-x, --p_thresh:  Significance cut off for t-dist based model." << std::endl;
	std::cerr << "                      " << "Lower values mean fewer pixels classified as WMH" << std::endl;
	std::cerr << "      " << "-d, --d_thresh:  Distance cut off for robust normal (M-Estimator) model." << std::endl;
	std::cerr << "                      " << "Higher values mean fewer pixels classified as WMH" << std::endl;
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	
	std::cerr << "Train new RF model mode:" << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-t, --train: training file name (.csv)"  << std::endl;
	std::cerr << "      " << "--sample,: Number of training samples"  << std::endl;
	std::cerr << "      " << "-m, --rf_mod: Specifies the .xml file name and path, where the Random Forest model will be saved/loaded."<< std::endl;
	std::cerr << std::endl;
	
	std::cerr << "=============================================================================="<< std::endl;
	
	std::cerr << "Outputing feature vectors:" << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-w, --wm_strip: Specifies the .nii file name and path of the WM stripped image."<< std::endl;
	std::cerr << "      " << "-o, --out: Specifies the (.csv) where features saved" << std::endl;
	std::cerr << "      " << "-y, --bravo: Specifies the .nii file name and path of the Second mode image e.g BRAVO" << std::endl;  
	std::cerr << "      " << "-f, --wm_mask: Specifies the .nii file name and path of the WM mask image" << std::endl;
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << "Testing modes:" << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-z: Testing modes. //If 0 enters test mode//" << std::endl;
	std::cerr << std::endl;
	

}//end of ShowCommandLineHelp()


void getFeatureOut(std::string WMStrippedFilename, std::string BRAVOFilename,std::string WMMaskFilename, std::string quantResultFilename2,int numberOfFeatures ){
	std::cout << "opening file: " << WMStrippedFilename << std::endl;
	std::cout << "reading in image" << std::endl;

	ImagePointer inNifti=NiftiReader(WMStrippedFilename);
	ImagePointer inNifti2=NiftiReader(BRAVOFilename);
	ImagePointer inNiftiM=NiftiReader(WMMaskFilename);


	MaskFilterType::Pointer maskFilter = MaskFilterType::New();
	
	maskFilter->SetInput(inNifti2);
	maskFilter->SetMaskImage(inNiftiM);
    maskFilter->Update();
    
	//Mat featMat =  getFeatureVector(inNifti,inNifti2, 2000);
	Mat locMat =  getLocationVector(inNifti);
	Mat locMat2 =  getLocationVector(maskFilter->GetOutput());

	std::cout << "save out...." << std::endl;
			
			std::ofstream myfile;
			myfile.open(quantResultFilename2);
						
			//myfile << format(featMat, "CSV") << std::endl;
			myfile << format(locMat, "CSV") << std::endl;
			
			myfile.close();
            //myfile.close();
	std::cout << "DONE!" << std::endl;
	
	std::cout << "save out...." << std::endl;
			
			std::ofstream myfile2;
			myfile2.open(renamer(quantResultFilename2, "_bravo"));
						
			//myfile << format(featMat, "CSV") << std::endl;
			myfile2 << format(locMat2, "CSV") << std::endl;
			
			myfile2.close();
            //myfile.close();
	std::cout << "DONE!" << std::endl;
	
	
	
	
  
}


//FOR TEST: 'testingSamples' IS PASSED TO THIS METHOD ONLY WHEN TESTING & DEBUGGING. IN REAL CASES, THESE SAMPLES WILL BE EXTRACTED FROM THE INPUT IMAGES.
//ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount,char *rfSegOutFilename, Mat testingSamples)
ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount,std::string rfSegOutFilename)
{
	std::cout << "Performing WMH segmentation..." << std::endl;
	
	MarginateImage(WMModStripImg,5);                        //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.
	
	/* The image will be rescaled in order to discard the long tail seen in the image histogram (i.e. about 0.3 of the histogram, which contains informationless voxels).
	* Note: itk::BinaryThresholdImageFilter can be used instead, if it is required to save the thresholded indexes (as in the W2MHS toolbox).
	*/
	double threshold=0.3*GetHistogramMax(WMModStripImg,127);
	
	/* The following pipeline will be processed to classify all voxels of the input image in the form of a regression model (i.e. closer to -1: less likely to be a WMH voxel. closer to +1: more likely to be a WMH voxel.)
	*
	* 1.iterating through the thresholded voxels
	* 2.creating a patch of 5*5*5, for each voxel
	* 3.extracting a feature vector of 2000 features for each patch
	* 4.sending the patch feature vector to the created/loaded Random Forest classifier for predicting its label.
	* 5.saving the corresponding prediction for each voxels in an image (i.e. 'RFSegOut' image)
	*/
	
	WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
	itk::ImageRegionIterator<ImageType> inputIterator(WMModStripImg, WMModStripImg->GetRequestedRegion());
	
	//creating an output image (i.e. a segmentation image) out of the prediction results.
	ImagePointer RFSegOutImage=ImageType::New();
	ImagePointer RFSegOutImage2=ImageType::New();
	
	ImageType::IndexType outStartIdx;
	outStartIdx.Fill(0);
	
	ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();
	
	ImageType::RegionType outRegion;
	outRegion.SetSize(outSize);
	outRegion.SetIndex(outStartIdx);
	
	RFSegOutImage->SetRegions(outRegion);
	RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
	RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
	RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
	RFSegOutImage->Allocate();
   
	RFSegOutImage2->SetRegions(outRegion);
	RFSegOutImage2->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
	RFSegOutImage2->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
	RFSegOutImage2->SetOrigin(WMModStripImg->GetOrigin());
	RFSegOutImage2->Allocate();
   
   
	itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);
	itk::ImageRegionIterator<ImageType> RFSegOutIterator2(RFSegOutImage2, outRegion);
	
//   int patchCount=0;   //FOR TEST
	while(!inputIterator.IsAtEnd())
	{
//      if(patchCount >= testingSamples.size[0])//FOR TEST
//      {//FOR TEST
//         inputIterator.GoToEnd();//FOR TEST
//      }//FOR TEST
//      else//FOR TEST
//      {//FOR TEST
		if(inputIterator.Get() > threshold)                  //the voxels that are not within the threshold (i.e. 0.3 the input image histogram) will be discarded and the corresponding value of 0 will be saved in the segmentation image.
        {
			Mat testSample = Mat(1, featuresCount, CV_32FC1);   //the test samples (i.e. the feature vectors) will be passed to the regression model one by one.
		
			//THE FOLLOWING TWO LINES SHOULD BE UNCOMMENTED when dealing with the actual image, not with the .csv test file
			ImagePointer patch=GetPatch(WMModStripImg,inputIterator.GetIndex(), 5);      // Note: GetPatch() is used to iterate through all voxels of the image and then to create feature vectors for each voxel patch.
			CreatePatchFeatureVector(patch,testSample);                           //creates feature vector of the 'patch' and loads it into 'testSample'.
			
			////         CreatePatchFeatureVector(patch,testSample,"<path-to-output-feature-file>",<label>); //FOR CREATING A NEW TRAINING DATA SET
			
			//            testSample=testingSamples.row(patchCount);//FOR TEST
		

			float prediction=RFRegressionModel->predict(testSample, Mat());
				RFSegOutIterator.Set(std::abs(prediction));
			
			if(prediction >=0)
			{
				RFSegOutIterator2.Set(1.0);
			}else
			{
				RFSegOutIterator2.Set(0.0);
			}
				//std::cout << "RF OUT: " << prediction << std::endl;

	//            if(patchCount % 500 == 0)   //FOR TEST
	//               std::cout << "Done segmenting patch# " << patchCount << "." << std::endl; //FOR TEST
	//
	//            ++patchCount; //FOR TEST
		}
		else
		{
			RFSegOutIterator.Set(0);   // thresholded voxels of the input image are set to 0 in the output segmentation image.
	
		}
		
		++RFSegOutIterator2;
		++RFSegOutIterator;
		++inputIterator;
//      }//FOR TEST
	}//end of iteration through the marginated and thresholded input image

	//saving the Random Forest classifier output image into a .nii file, provided as a command line argument
	NiftiWriter(RFSegOutImage,rfSegOutFilename);
	NiftiWriter(RFSegOutImage2, renamer(rfSegOutFilename,"_h")); //hard classify test
	//FOR TEST & DEBUG
	//   std::cout << "Calculating testing data set error..." << std::endl;
	//   CalculateMSE(RFSegOutImage,NiftiReader("<path-to-reference-RFSegOut-nifti-image>"));
	
	std::cout << "Done WMH segmentation successfully." << std::endl;
	return RFSegOutImage;
}//end of ClassifyWMHs()


// new method for seg may 2016
ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,std::string rfSegOutFilename)
{
   std::cout << "Performing WMH segmentation using new method..." << std::endl;

   MarginateImage(WMModStripImg,5);                        //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.
   WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> inputIterator(WMModStripImg, WMModStripImg->GetRequestedRegion());

   //creating an output image (i.e. a segmentation image) out of the prediction results.
   ImagePointer RFSegOutImage=ImageType::New();
   
   ImageType::IndexType outStartIdx;
   outStartIdx.Fill(0);

   ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();

   ImageType::RegionType outRegion;
   outRegion.SetSize(outSize);
   outRegion.SetIndex(outStartIdx);

   RFSegOutImage->SetRegions(outRegion);
   RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
   RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
   RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
   RFSegOutImage->Allocate();
   
   
   
   itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);

//   int patchCount=0;   //FOR TEST
   while(!inputIterator.IsAtEnd())
   {

	
		 
         ++RFSegOutIterator;
         ++inputIterator;

   }//end of iteration through the marginated and thresholded input image

   //saving the Random Forest classifier output image into a .nii file, provided as a command line argument
   NiftiWriter(RFSegOutImage,rfSegOutFilename);

   std::cout << "Done WMH segmentation successfully." << std::endl;
   return RFSegOutImage;
}//end of ClassifyWMHs()

//adds suffix and puts extensiuon back in place
std::string renamer(std::string baseName, std::string suffix){
	
	size_t lastindex = baseName.find_last_of("."); 
	std::string rawname = baseName.substr(0, lastindex);
	std::string exten = baseName.substr(baseName.find_last_of(".") + 1);
	
	std::string newName = rawname + suffix +"."+ exten;
	
	return newName;

}




double phi(double d){
	double c = 1.345;
	double temp = 0.0;
		
		
	if(abs(d) < c) {
		temp = d;
	}else{
		temp= c*sgn(d);
	}
		
		
	return temp/d; 
}


 template <typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
}

double df_eq_est(double v, double v_sum){
	 return -Digamma(0.5*v) + std::log(0.5*v) + v_sum + Digamma((v+1.)/2.) - std::log((v+1.)/2.)+1.;
}

std::string	vecToString(vector<string> v){
		std::string s;
		for (const auto &piece : v) s += piece;
		return s;
	}

double Digamma(double x){
	
	static const double dg_coeff[10] = 
	{-.83333333333333333e-1, .83333333333333333e-2,-.39682539682539683e-2,
  .41666666666666667e-2,-.75757575757575758e-2, .21092796092796093e-1,
 -.83333333333333333e-1, .4432598039215686    ,-.3053954330270122e+1, .125318899521531e+2};
 
	   //from http://perso.ensta-paristech.fr/~lunevill/doc_melina++_v039/src/special_functions/GammaFunctions.c++.html
		// not worried about complex, int, int+.5 cases
		double dgam = 0.0;
   
       //Use formula for derivative of LogGamma(z)
      if ( x < 10 )
      {
         int n = 10-int(x);
          //for | x | < 10 , use recursively DiGamma(x) = DiGamma(x+1) - 1/x
         for ( int k = 0; k < n; k++ ) dgam -= 1./(k+x);
         x += n;
      }
      double overx2 = (1./(x*x));
      double  overx2k = overx2;
      dgam += log(x) - .5/x;
      
      for (int k = 0; k < 10; k++, overx2k *= overx2 ) dgam += dg_coeff[k]*overx2k;
   
   return dgam;
}




//FOR TEST: 'testingSamples' IS PASSED TO THIS METHOD ONLY WHEN TESTING & DEBUGGING. IN REAL CASES, THESE SAMPLES WILL BE EXTRACTED FROM THE INPUT IMAGES.
//ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount,char *rfSegOutFilename, Mat testingSamples)
ImagePointer ClassifyWMHsT(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double p_thresh_const, double a, double b)
{
//alg parameters
	int max_iter=100;
	double tol = 0.0001;
	double v_init =4;
		

   std::cout << "Performing WMH segmentation t-dist ..." << std::endl;

   MarginateImage(WMModStripImg,5);                        
   
   //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.

   /* The image will be rescaled in order to discard the long tail seen in the image histogram (i.e. about 0.3 of the histogram, which contains informationless voxels).
    * Note: itk::BinaryThresholdImageFilter can be used instead, if it is required to save the thresholded indexes (as in the W2MHS toolbox).
    */
   
   WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
   //itk::ImageRegionIterator<ImageType> inputIterator(WMModStripImg, WMModStripImg->GetRequestedRegion());

   //creating an output image (i.e. a segmentation image) out of the prediction results.
   ImagePointer RFSegOutImage=ImageType::New();

   ImageType::IndexType outStartIdx;
   outStartIdx.Fill(0);

   ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();
   ImageType::RegionType outRegion;
   outRegion.SetSize(outSize);
   outRegion.SetIndex(outStartIdx);

   RFSegOutImage->SetRegions(outRegion);
   RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
   RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
   RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
   RFSegOutImage->Allocate();
  
	
	//creating an output image (i.e. a segmentation image) out of the prediction results.
     
   NeighborhoodIteratorType::RadiusType radius;
   radius.Fill(1);
    
   // set up iterators
    itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);
 
	
    NeighborhoodIteratorType inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
  	
  	std::vector<double> y;
  	std::vector<double> zeros;
  	//model fit
  	// robust fit normal dist with M estimator
  	
	while(!inputIterator.IsAtEnd())
	{
		y.push_back(inputIterator.GetCenterPixel());
		if(inputIterator.GetCenterPixel()>0.0){
			zeros.push_back(1.0);
		}else{
			zeros.push_back(0.0);
		}
		++inputIterator;
	}
	
	
	vecPair model = fitTdist( y,  zeros,   tol,  max_iter,  a,  b);
	
	
	double mu = model.second.at(2);
	double sigma = model.second.at(1);   
	double v = model.second.at(0);
	
	//std::cout << "df  " << v << " mean "<< mu << " sd  "<< sigma << std::endl;
	
	double p_thresh_const_2 = 1.0-p_thresh_const;
		
	// Construct a students_t distribution with v degrees of freedom:
    students_t d1(v);
    std::vector<double> q(y.size(), 0.0);
    std::transform(y.begin(), y.end(), q.begin(),  [d1, mu, sigma](double x) { return cdf(d1, (x-mu)/std::sqrt(sigma)); }); 
	
  	vector<double>::iterator it; 
    it=q.begin();
	   
	inputIterator.GoToBegin();
	
	int c = (inputIterator.Size());
	int mid = (inputIterator.Size()) / 2;
	
	
	
	while(!inputIterator.IsAtEnd())
	{
		double acc = 0.0;
		double neighbourhood_mean = 0.0;
		int non_zero = 0;
		
		for(int i=0; i<c; i++){		
			double temp  = inputIterator.GetPixel(i);
			if(i != (c/2))
			{
				acc += temp;
				if(temp>0){++non_zero;}
			}
		
		}	
		neighbourhood_mean = acc/non_zero;			       
        
         if(neighbourhood_mean > mu & *it > (1.- p_thresh_const)){
			RFSegOutIterator.Set(1.0);
		 }else{
			RFSegOutIterator.Set(0.0);
		 }
           
		 		 
		 ++RFSegOutIterator;
		 ++inputIterator;
		 it++;
	}
	
	
	ImagePointer RFSegOutImage2=ImageType::New();
	RFSegOutImage2=RFSegOutImage;
	NeighborhoodIteratorType RFSegOutIterator2(radius, RFSegOutImage2, outRegion);
	
	RFSegOutIterator.GoToBegin();
	//adjust using counts
	
	
	while(!RFSegOutIterator2.IsAtEnd())
	{
	   
		int count = 0;
		
		for(int i=0; i<c; i++){ //don't count centre pixel
			if(RFSegOutIterator2.GetPixel(i) == 1.0 & i != (c/2) ){++count;}
					
		}	
		
		if(count< min_neighbour){
			RFSegOutIterator.Set(0.0);
		}else{
			RFSegOutIterator.Set(RFSegOutIterator2.GetCenterPixel());
			}
			
  	   
		
		++RFSegOutIterator;
		++RFSegOutIterator2;
	}
		
  
   NiftiWriter(RFSegOutImage,rfSegOutFilename.c_str()); 

   std::cout << "Done WMH segmentation successfully." << std::endl;
   return RFSegOutImage;
}//end of ClassifyWMHs()


//FOR TEST: 'testingSamples' IS PASSED TO THIS METHOD ONLY WHEN TESTING & DEBUGGING. IN REAL CASES, THESE SAMPLES WILL BE EXTRACTED FROM THE INPUT IMAGES.
//ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount,char *rfSegOutFilename, Mat testingSamples)
ImagePointer ClassifyWMHsM(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double out_thresh_const, int max_iter=100, double tol = 0.0001)
{

	
	std::cout << "Performing WMH segmentation with M-estimator..." << std::endl;
	
	MarginateImage(WMModStripImg,5);                        
	//Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.
	// kept in to keep consistent with older RF model
	
	
	//creating an output image (i.e. a segmentation image) out of the prediction results.
	WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
	ImagePointer RFSegOutImage=ImageType::New();
	
	ImageType::IndexType outStartIdx;
	outStartIdx.Fill(0);
	
	ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();
	ImageType::RegionType outRegion;
	outRegion.SetSize(outSize);
	outRegion.SetIndex(outStartIdx);
	
	RFSegOutImage->SetRegions(outRegion);
	RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
	RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
	RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
	RFSegOutImage->Allocate();
	
	     
	NeighborhoodIteratorType::RadiusType radius;
	radius.Fill(1);
	
	// set up iterators
	itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);
		
	NeighborhoodIteratorType inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
	
	//initialise vectors
	std::vector<double> y;
	std::vector<double> zeros;
	
	//model fit
	// robust fit normal dist with M estimator
  	
  	//put data into std::vector
	while(!inputIterator.IsAtEnd())
	{
		y.push_back(inputIterator.GetCenterPixel());
		if(inputIterator.GetCenterPixel()>0.0){
			zeros.push_back(1.0);
		}else{
			zeros.push_back(0.0);
		}
		++inputIterator;
	}
	
	
	// fit model
	vecPair model = fitMest(y, zeros,  tol, max_iter);
		
	double mu = model.second.at(1);
	double sigma = model.second.at(0);   
	
	vector<double>::iterator it; 
	
    it=model.first.begin();
    
	//malahabnois dist cutoff as in van leemput 99
	double outly = -2*std::log(out_thresh_const*std::sqrt(2*3.141593*sigma)); 
    
    
   
	inputIterator.GoToBegin();
	int c = (inputIterator.Size());
	int mid = (inputIterator.Size()) / 2;
	
	
	
	//main loop
	
   while(!inputIterator.IsAtEnd())
   {
		double acc = 0.0;
		double neighbourhood_mean = 0.0;
		int non_zero = 0;
		
		
		//get neighbourhood mean
		for(int i=0; i<c; i++){		
			double temp  = inputIterator.GetPixel(i); 
			
			if(i != (c/2))
			{
				acc += temp;
				if(temp>0){++non_zero;}
			}
		
		}	
		
		if(non_zero>0.){
			neighbourhood_mean = acc/non_zero;			       
        }else{
			neighbourhood_mean = 0.;	
		}
           
       
        
         
		
		 //if neighbourhood mean > mean and point above cut off set as on
         if(neighbourhood_mean > mu & *it > outly){
			RFSegOutIterator.Set(1.0);
			std::cout << "local mean " << neighbourhood_mean << " mean "<< mu << " dist "<< *it << " cutoff "<< outly  << std::endl;
			
		 }else{
			RFSegOutIterator.Set(0.0);
		 }
           
		 		 
		 ++RFSegOutIterator;
		 ++inputIterator;
		 it++;
	}
		
	
	ImagePointer RFSegOutImage2=ImageType::New();
	RFSegOutImage2=RFSegOutImage;
	NeighborhoodIteratorType RFSegOutIterator2(radius, RFSegOutImage2, outRegion);
	
	RFSegOutIterator.GoToBegin();
	
	//loop again
	//adjust using counts
	while(!RFSegOutIterator2.IsAtEnd())
	{
	   
		int count = 0;
		
		for(int i=0; i<c; i++){
			if(RFSegOutIterator2.GetPixel(i) == 1.0 & i != (c/2)){++count;}
		}	
		
		// cancel when min neigbours too low
		if(count< min_neighbour){
			RFSegOutIterator.Set(0.0);
		}else{
			RFSegOutIterator.Set(RFSegOutIterator2.GetCenterPixel());
		}
			
  	   
		
		++RFSegOutIterator;
		++RFSegOutIterator2;
	}
		
  
   NiftiWriter(RFSegOutImage,rfSegOutFilename.c_str()); 

   std::cout << "Done WMH segmentation successfully." << std::endl;
   return RFSegOutImage;
}//end of ClassifyWMHs()



ImagePointer ClassifyWMHsMRF(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double out_thresh_const, int max_iter=100, double tol = 0.0001)
{

	
	std::cout << "Performing WMH segmentation with M-estimator + MRF ..." << std::endl;
	
	MarginateImage(WMModStripImg,5);                        
	//Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.
	// kept in to keep consistent with older RF model
	
	
	//creating an output image (i.e. a segmentation image) out of the prediction results.
	WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
	ImagePointer RFSegOutImage=ImageType::New();
	
	ImageType::IndexType outStartIdx;
	outStartIdx.Fill(0);
	
	ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();
	ImageType::RegionType outRegion;
	outRegion.SetSize(outSize);
	outRegion.SetIndex(outStartIdx);
	
	RFSegOutImage->SetRegions(outRegion);
	RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
	RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
	RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
	RFSegOutImage->Allocate();
	
	     
	NeighborhoodIteratorType::RadiusType radius;
	radius.Fill(1);
	
	// set up iterators
	itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);
		
	NeighborhoodIteratorType inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
	
	//initialise vectors
	std::vector<double> y;
	std::vector<double> zeros;
	
	//model fit
	// robust fit normal dist with M estimator
  	
  	//put data into std::vector
	while(!inputIterator.IsAtEnd())
	{
		y.push_back(inputIterator.GetCenterPixel());
		if(inputIterator.GetCenterPixel()>0.0){
			zeros.push_back(1.0);
		}else{
			zeros.push_back(0.0);
		}
		++inputIterator;
	}
	
	
	// fit model
	vecPair model = fitMest(y, zeros,  tol, max_iter);
		
	double mu = model.second.at(1);
	double sigma = model.second.at(0);   
	
	vector<double>::iterator it; 
	
    it=model.first.begin();
    
	//malahabnois dist cutoff as in van leemput 99
	double outly = -2*std::log(out_thresh_const*std::sqrt(2*3.141593*sigma)); 
    
    
   
	inputIterator.GoToBegin();
	int c = (inputIterator.Size());
	int mid = (inputIterator.Size()) / 2;
	
	
	
	//main loop
	
   while(!inputIterator.IsAtEnd())
   {
     	//if neighbourhood mean > mean and point above cut off set as on
         if(*it > outly){
			RFSegOutIterator.Set(1.0);
			//std::cout << "local mean " << neighbourhood_mean << " mean "<< mu << " dist "<< *it << " cutoff "<< outly  << std::endl;

		 }else{
			RFSegOutIterator.Set(0.0);
		 }
           
		 		 
		 ++RFSegOutIterator;
		 ++inputIterator;
		 it++;
	}
		
	
	
	
	int numberOfClasses =2 ;
	int numberOfIterations = 10;
	double errTolMRF = 1e-7;
	double smoothingFactor  = 2.0;
	typedef itk::MRFImageFilter< ImageType, ImageType> MRFFilterType;
	MRFFilterType::Pointer mrfFilter = MRFFilterType::New();
	
	
	mrfFilter->SetInput( RFSegOutImage);
	mrfFilter->SetNumberOfClasses( numberOfClasses );
	mrfFilter->SetMaximumNumberOfIterations( numberOfIterations );
	mrfFilter->SetErrorTolerance( errTolMRF  );
	mrfFilter->SetSmoothingFactor( smoothingFactor );
	mrfFilter->SetNeighborhoodRadius( 1 );
	
	std::vector< double > weights;
	weights.push_back(0.5);
	weights.push_back(0.5);
	weights.push_back(0.5);
	weights.push_back(0.5);
	weights.push_back(1.0); // This is the central pixel
	weights.push_back(0.5);
	weights.push_back(0.5);
	weights.push_back(0.5);
	weights.push_back(0.5);
	  
	mrfFilter->SetMRFNeighborhoodWeight( weights );
	  
	typedef MRFFilterType::OutputImageType ImageType;
	
	typedef itk::ImageClassifierBase< ImageType, ImageType >   SupervisedClassifierType;
	SupervisedClassifierType::Pointer classifier = SupervisedClassifierType::New();
	
	typedef itk::Statistics::MinimumDecisionRule DecisionRuleType;
	DecisionRuleType::Pointer  classifierDecisionRule = DecisionRuleType::New();
	
	classifier->SetDecisionRule( classifierDecisionRule.GetPointer() );
	
	typedef itk::Statistics::DistanceToCentroidMembershipFunction<std::vector < float > > MembershipFunctionType; 
	typedef MembershipFunctionType::Pointer MembershipFunctionPointer;
	//typedef itk::Statistics::ManhattanDistanceMetric< float > distanceMetricType;
	
	//distanceMetricType::New() distanceMetric;
	//distanceMetric->SetOrigin(0.0);
	
	double meanDistance = 0;
	MembershipFunctionType::CentroidType centroid(1);
	
		//~ 
	//~ MembershipFunctionPointer membershipFunction0 =  MembershipFunctionType::New();
	//~ //membershipFunction0->SetDistanceMetric(distanceMetric);
	//~ centroid[0] = 0.0;
	//~ membershipFunction0->SetCentroid( centroid );
	//~ classifier->AddMembershipFunction( membershipFunction0 );
		//~ 
	//~ MembershipFunctionPointer membershipFunction1 =  MembershipFunctionType::New();
	//~ //membershipFunction1->SetDistanceMetric(distanceMetric);
	//~ centroid[0] = outly*2;
	//~ membershipFunction1->SetCentroid( centroid );
	//~ classifier->AddMembershipFunction( membershipFunction1 );
		
	mrfFilter->Update();
	ImagePointer RFSegOutImage2=ImageType::New();
	RFSegOutImage2 = mrfFilter->GetOutput();
	
	
	NeighborhoodIteratorType RFSegOutIterator2(radius, RFSegOutImage2, outRegion);
	RFSegOutIterator.GoToBegin();
	
	//loop again
	//adjust using counts
	while(!RFSegOutIterator2.IsAtEnd())
	{
	   
		int count = 0;
		
		for(int i=0; i<c; i++){
			if(RFSegOutIterator2.GetPixel(i) == 1.0 & i != (c/2)){++count;}
		}	
		
		// cancel when min neigbours too low
		if(count< min_neighbour){
			RFSegOutIterator.Set(0.0);
		}else{
			RFSegOutIterator.Set(RFSegOutIterator2.GetCenterPixel());
		}
			
  	   
		
		++RFSegOutIterator;
		++RFSegOutIterator2;
	}
		
  
   NiftiWriter(RFSegOutImage,rfSegOutFilename.c_str()); 

   std::cout << "Done WMH segmentation successfully." << std::endl;
   return RFSegOutImage;
}//end of ClassifyWMHs()



vecPair fitMest(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter){

	std::cout<< "observation length:  " << y.size() << " pixels" <<   std::endl;
	
  	//double z_sum = std::accumulate(zeros.begin(), zeros.end(), 0.0);
  	double y_sum = std::accumulate(y.begin(), y.end(), 0.0);
	//double y_mean = y_sum / y.size();
	
	//allocate vectors
	std::vector<double> w = zeros;  
	std::vector<double> d(y.size(), 0.0);
	std::vector<double> wy_prod(y.size(), 0.0);  
	std::vector<double> ymu_minus(y.size(), 0.0);
	std::vector<double> ymu_minus2(y.size(), 0.0);
	std::vector<double> sq_sum(y.size(), 0.0);
	std::vector<double> sq_sum_w(y.size(), 0.0);
	std::vector<double> w2(y.size(), 0.0); 
	std::vector<double> diff_w(y.size(), 0.0);
	std::vector<double> outliness(y.size(), 0.0);
	std::vector<double> log_w(y.size(), 0.0);
	
	//allocate double
	double mu = 0.0;
	double sigma = 0.0;
	double wy_sum = 0.0;
	double w_sum = 0.0;
	double w_sq_sum = 0.0;
	double v_sum = 0.0;
	double diff = 1.0;
	
	//allocated int
	int k =0;
	
	
	//iterative fit for M-est
	while(diff>tol & k <max_iter ){
		std::transform(w.begin(), w.end(), zeros.begin(), w.begin(), std::multiplies<double>());
		
		std::transform(y.begin(), y.end(), w.begin(), wy_prod.begin(), std::multiplies<double>()); 
		wy_sum = std::accumulate(wy_prod.begin(), wy_prod.end(), 0.0);
		w_sum = std::accumulate(w.begin(), w.end(), 0.0);
		
		mu = y_sum/w_sum;
		
		std::transform(y.begin(), y.end(), ymu_minus.begin(), [mu](double x) { return x - mu; });
			
		std::transform(ymu_minus.begin(), ymu_minus.end(), ymu_minus.begin(), sq_sum.begin(), std::multiplies<double>()); 
		std::transform(sq_sum.begin(), sq_sum.end(), w.begin(), sq_sum_w.begin(), std::multiplies<double>()); 
		sigma =  std::accumulate(sq_sum_w.begin(), sq_sum_w.end(), 0.0) / w_sum; //sigma is variance not sd
		
		std::transform(sq_sum.begin(), sq_sum.end(), d.begin(), [sigma](double x) { return std::sqrt(x / sigma); });
		
		w2 = w;
		std::transform(d.begin(), d.end(), w.begin(), phi);
		std::transform(w.begin(), w.end(), zeros.begin(), w.begin(), std::multiplies<double>()); //ignores pixels with value = 0
		
		std::transform(w.begin(), w.end(), w2.begin(), diff_w.begin(), [](double x, double y) {return std::abs(x-y);}); 
		
		diff = std::accumulate(diff_w.begin(), diff_w.end(), 0.0);
		++k;
	
	}
	
	std::transform(d.begin(), d.end(), zeros.begin(), d.begin(), std::multiplies<double>());  //ignores pixels with value = 0
	
	std::cout<< "iter " << k << " mu " << mu << " sigma " << sigma << " diff " << diff << std::endl;
	
	std::vector<double> params;
	params.push_back(sigma);
	params.push_back(mu);
	
	
	vecPair returnValues = vecPair(d, params);
	
	return(returnValues);
}



vecPair fitTdist(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter, double a, double b){

	std::cout<< "observation length:  " << y.size() << " pixels" <<   std::endl;
	
  	double z_sum = std::accumulate(zeros.begin(), zeros.end(), 0.0);
  	double y_sum = std::accumulate(y.begin(), y.end(), 0.0);
	//double y_mean = y_sum / y.size();
	
	//allocate vectors
	std::vector<double> w = zeros;  
	std::vector<double> d(y.size(), 0.0);
	std::vector<double> wy_prod(y.size(), 0.0);  
	std::vector<double> ymu_minus(y.size(), 0.0);
	std::vector<double> ymu_minus2(y.size(), 0.0);
	std::vector<double> sq_sum(y.size(), 0.0);
	std::vector<double> sq_sum_w(y.size(), 0.0);
	std::vector<double> w2(y.size(), 0.0); 
	std::vector<double> diff_w(y.size(), 0.0);
	std::vector<double> outliness(y.size(), 0.0);
	std::vector<double> log_w(y.size(), 0.0);
	
	//allocate double
	double mu = 0.0;
	double sigma = 0.0;
	double wy_sum = 0.0;
	double w_sum = 0.0;
	double w_sq_sum = 0.0;
	double v_sum = 0.0;
	double diff = 1.0;
	double v = 4.0; // initial guess for degrees of freedom, could be user param but actually not very important, range (a,b) more important.
	
	//allocated int
	int k =0;
	
	while(diff>tol & k <max_iter ){
		    
		boost::uintmax_t df_max_iter=500; // boost solver params, could be user params but relatively unimportant
		tools::eps_tolerance<double> tol(30);
		
		std::transform(y.begin(), y.end(), w.begin(), wy_prod.begin(), std::multiplies<double>()); 
		wy_sum = std::accumulate(wy_prod.begin(), wy_prod.end(), 0.0);
		w_sum = std::accumulate(w.begin(), w.end(), 0.0);
		mu = y_sum/z_sum;
		
		std::transform(y.begin(), y.end(), ymu_minus.begin(), [mu](double x) { return x - mu; });
			
		
		std::transform(ymu_minus.begin(), ymu_minus.end(), ymu_minus.begin(), sq_sum.begin(), std::multiplies<double>()); 
		std::transform(sq_sum.begin(), sq_sum.end(), w.begin(), sq_sum_w.begin(), std::multiplies<double>()); 
		sigma =  std::accumulate(sq_sum_w.begin(), sq_sum_w.end(), 0.0) / z_sum; //sigma is variance not sd
		
				
				
		std::transform(w.begin(),w.end(), log_w.begin(), [](double x) { return (std::log(x) - x); });
		std::transform(log_w.begin(), log_w.end(), log_w.begin(), [](double x) { return isinf(x) ? 0.0 : x  ; }); //ignores pixels with value = 0
		v_sum = std::accumulate(log_w.begin(), log_w.end(), 0.0)/ z_sum;
		
		if(k !=0){
			df_eq_func rootFun = df_eq_func(v_sum);
			std::pair<double, double>  r1= tools::bisect(rootFun, a, b, tol, df_max_iter);
			v = (r1.first + r1.second)/2.0; 
		}
		
		
		std::transform(sq_sum.begin(), sq_sum.end(), d.begin(), [sigma](double x) { return std::sqrt(x / sigma); });
		w2 = w;	
		
		
		std::transform(d.begin(), d.end(), w.begin(), [v](double x) { return ((v+1) / (v+x*x));});
		std::transform(w.begin(), w.end(), zeros.begin(), w.begin(), std::multiplies<double>()); //ignores pixels with value = 0
		
		std::transform(w.begin(), w.end(), w2.begin(), diff_w.begin(), [](double x, double y) {return std::abs(x-y);}); 
		
		diff = std::accumulate(diff_w.begin(), diff_w.end(), 0.0);
		
		//std::cout<< "iter " << k << " mu " << mu << " sigma " << sigma << " v " << v << " diff " << diff << std::endl;
		
		++k;
	
	}
    std::transform(d.begin(), d.end(), zeros.begin(), d.begin(), std::multiplies<double>());  //ignores pixels with value = 0

	std::cout<< "iter " << k << " mu " << mu << " sigma " << sigma << " v " << v << " diff " << diff << std::endl;	

	std::vector<double> params;
	params.push_back(v);
	params.push_back(sigma);
	params.push_back(mu);
	
	vecPair returnValues = vecPair(d, params);
	return(returnValues);
}



ImagePointer CreateWMHPmap(ImagePointer rfSegmentedImg, ImagePointer refImg, std::string gcFilename, std::string pmapFilename)
{
   std::cout << "Creating WMH probability map..." << std::endl;

   ImagePointer unrectifiedPmap;
   ImagePointer invertedGcInput;
   try
   {
      /* ventricle PVE..."It is observed that several regions lying on the boundaries of ventricles are miss-segmented as
       * GM and/or CSF. Hence, a ventricular template has been extracted from CSF PVE to adjust the ROI to include these
       * periventricular regions.
       */
      invertedGcInput=InvertImage(NiftiReader(gcFilename),1);      // As the 'GMCSFstrip' is a binary one, we set the maximum of the 'InvertImage' function to '1'.

      //unrectifiedPmap contains noisy detections as well.
      unrectifiedPmap=MultiplyTwoImages(rfSegmentedImg,MinMaxNormalisation(refImg));      //min-max normalisation of the refImg (WM_modstrip)
   }
   catch(itk::ExceptionObject &)
   {
      std::cerr << "Failed to create the WMH probability map: either the GMCSF image or the RFSegOut image couldn't be loaded properly!" << std::endl;
   }

   //cleaning up noisy and small detections

   unrectifiedPmap->SetRequestedRegionToLargestPossibleRegion();
   ImageType::SizeType sizes = unrectifiedPmap->GetLargestPossibleRegion().GetSize();
 
   
   
   if((sizes[0]*sizes[1]*sizes[2])<10000){ // this technique seems to go weird on my small test files
			   NiftiWriter(unrectifiedPmap,pmapFilename);   //'unrectifiedPmap' is a rectified pmap at this stage

		}

   
   itk::ImageRegionIterator<ImageType> pmapIterator(unrectifiedPmap, unrectifiedPmap->GetRequestedRegion());

   //Note: the .Set() method of ImageSliceIteratorWithIndex didn't work. Otherwise, that type of iteration is a better match and probably more efficient method of iteration here.
   for(int dim3=0; dim3 < rfSegmentedImg->GetLargestPossibleRegion().GetSize()[2] && !pmapIterator.IsAtEnd() ; ++dim3)   //iterate through the segmented image slice by slice, along the 3rd dimension
   {
      ImagePointer2D gcSliceBoundaris=Get2DBinaryObjectsBoundaries(Get2DSlice(invertedGcInput,2,dim3));      //as we would like to get slices of the 'gc' image along the Z-axis, we set the 'plane' to 2.
      if(FindIndicesByIntensity(gcSliceBoundaris,1).size() != 0)   //if there aren't any objects in the current slice of 'gc', then continue
      {
         //Note: intensity values of greater than 0.5 in 'pmap' represent WMHs, and intensity values of less than 0.5 represent non-WMHs
         int objectCount;   //represents the number of objects found in the current slice of the pmap image
         ImagePointer2D labelledImg=LabelBinaryObjects(BinarizeThresholdedImage(Get2DSlice(unrectifiedPmap,2,dim3),0.5,1),&objectCount);

         /** according to W2MHS, any objects (in the current slice), which are:
          *       1) only 1 voxel wide
          *       or
          *       2) if they're smaller than 100 voxels AND more than half of their voxels are closer than a threshold (i.e. 'clean_th'=2.5) to the 'gc' region
          * should be considered as a noisy detection. That is, the object's voxel values will be set to 0.
          *
          * a Note for "closer than": the "shortest" Euclidean distance of each voxel of the object from all voxels of all objects of 'gc' in the corresponding slice is calculated.
          **/
         for(int obj=1; obj <= objectCount ; ++obj)
         {
            //iterate over the labelledImg only in the region with the intensity value equal to obj
            std::vector<ImageType2D::IndexType> currentObjIndices=FindIndicesByIntensity(labelledImg,obj);
            int currentObjVoxelCount=currentObjIndices.size();

            //calculate the min distance of each object's voxel from all objects' voxels of the corresponding 'gc' slice
            int noisyDetectionsCount=0;
            if(currentObjVoxelCount != 1)
            {
               for (std::vector<ImageType2D::IndexType>::iterator idxIterator = currentObjIndices.begin(); idxIterator != currentObjIndices.end(); ++idxIterator)
               {
                  double minDistance=GetMinIndexDistanceFromObjects(*idxIterator,gcSliceBoundaris);
                  if(minDistance <= 2.5)   //2.5 is the clean_th hard-coded in W2MHS TODO: should come from user?
                     ++noisyDetectionsCount;
               }
            }
            //
            if(currentObjVoxelCount == 1 || (currentObjVoxelCount < 100 && noisyDetectionsCount/currentObjVoxelCount > 0.5))
            {
               for (std::vector<ImageType2D::IndexType>::iterator idxIterator = currentObjIndices.begin(); idxIterator != currentObjIndices.end(); ++idxIterator)
               {
                  ImageType::IndexType index3D;
                  index3D[0]=(*idxIterator)[0];
                  index3D[1]=(*idxIterator)[1];
                  index3D[2]=dim3;
                  pmapIterator.SetIndex(index3D);
                  pmapIterator.Set(0);
               }
            }
         }//end of for iterating through objects of the pmap slice
      }//end of if checking whether there are any objects in the current slice of 'gc'
   }//end of iterating through slices of 'rfSegmentedImg'
   
   
   std::cout << "WMH probability map is saved into: " << pmapFilename << std::endl;
   
    if((sizes[0]*sizes[1]*sizes[2])>=10000){
       NiftiWriter(unrectifiedPmap,pmapFilename);   //'unrectifiedPmap' is a rectified pmap at this stage
    }
   
   
   return unrectifiedPmap;
}//end of CreateWMHPmap()




void QuantifyWMHs(float pmapCut, ImagePointer pmapImg, std::string ventricleFilename, std::string outputFilename)
{
   std::cout << "Quantifying WMH segmentations..." << std::endl;

   //voxelResolution has been hard-coded as '0.5' in the W2MHS toolbox code ('V_res').
   //float voxelRes=pmapImg->GetSpacing()[0];
   float voxelRes=pmapImg->GetSpacing()[0]*pmapImg->GetSpacing()[1]*pmapImg->GetSpacing()[2];
   float dist =  pmapImg->GetSpacing()[0];//assumes same in all dim.
   int distDP=8;   //According to the definition of "Periventricular Region" in "“Anatomical mapping of white matter hyper- intensities (wmh) exploring the relationships between periventricular wmh, deep wmh, and total wmh burden, 2005", voxels closer than 8mm to ventricle are considered as 'Periventricular' regions.
   int k=1;   // Note: "EV calculates the hyperintense voxel count 'weighted' by the corresponding likelihood/probability, where 'k' controls the degree of weight (in formula (2))." ... 'k' is called 'gamma' in the W2MHS toolbox code.

   ImagePointer thresholdedPmap=ThresholdImage(pmapImg,pmapCut,1);
   float EV=voxelRes * SumImageVoxels(PowerImageToConst(thresholdedPmap,k));

   //The following block differentiates the Periventricular from Deep hyperintensities.
   ImagePointer ventricle=NiftiReader(ventricleFilename.c_str());

   //creating a 'Ball' structuring element for dilating the ventricle area. (It can also be created using 'itk::BinaryBallStructuringElement'.)
   /*
    * "A Neighborhood has an N-dimensional radius. The radius is defined separately for each dimension as the number of pixels that the neighborhood extends outward from the center pixel.
    * For example, a 2D Neighborhood object with a radius of 2x3 has sides of length 5x7. However, in the case of balls and annuli, this definition is slightly different from the parametric
    * definition of those objects. For example, an ellipse of radius 2x3 has a diameter of 4x6, not 5x7. To have a diameter of 5x7, the radius would need to increase by 0.5 in each dimension.
    * Thus, the "radius" of the neighborhood and the "radius" of the object should be distinguished.
    * To accomplish this, the "ball" and "annulus" structuring elements have an optional flag called "radiusIsParametric" (off by default). Setting this flag to true will use the parametric definition
    * of the object and will generate structuring elements with more accurate areas, which can be especially important when morphological operations are intended to remove or retain objects of particular sizes.
    * When the mode is turned off (default), the radius is the same, but the object diameter is set to (radius*2)+1, which is the size of the neighborhood region. Thus, the original ball and annulus structuring
    * elements have a systematic bias in the radius of +0.5 voxels in each dimension relative to the parametric definition of the radius. Thus, we recommend turning this mode on for more accurate structuring elements,
    * but this mode is turned off by default for backward compatibility."
    */
   typedef itk::FlatStructuringElement<3> StructuringElement3DType;      //3 is the image dimension
   StructuringElement3DType::RadiusType radius3D;
   radius3D.Fill(distDP/dist);      //"voxels closer than 8mm to ventricle are considered as 'DeepPeriventricular' regions.". Thus, when for example the voxel resolution of the image is 2mm, it means that those voxels that are within 8mm/2mm=4 voxels away from the ventricle should be considered as periventricular.
   StructuringElement3DType structuringElem=StructuringElement3DType::Ball(radius3D);
//   structuringElem.RadiusIsParametricOn();   //and "structuringElem.SetRadiusIsParametric(true)" make no difference in calculations, decpite what's been claimed (see above explanation)!

   //dilating the ventricle area by the 'Ball' structuring element
   float pEV=0;
   float dEV=0;
   try
   {
      typedef itk::BinaryDilateImageFilter<ImageType, ImageType, StructuringElement3DType> BinaryDilateImageFilterType;
      BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
      dilateFilter->SetInput(ventricle);
      dilateFilter->SetKernel(structuringElem);
      dilateFilter->SetForegroundValue(1);   //Note: "Set the value in the image to consider as "foreground". Defaults to maximum value of PixelType."
      dilateFilter->SetBackgroundValue(0);   //      Thus, these two lines are necessary, as the index type is set to float.
      ImagePointer dilatedVentricle=dilateFilter->GetOutput();
      dilatedVentricle->Update();
		
		
	   //string coolTest = "/data/home/uqajon14/Output/Jvent_dilate_test.nii";
	   //bool test = NiftiWriter(dilatedVentricle,coolTest);
      //separating Deep and Periventricular areas in pmap.
      ImagePointer periventricularPmap=MultiplyTwoImages(thresholdedPmap,dilatedVentricle);
      ImagePointer deepPmap=MultiplyTwoImages(thresholdedPmap,InvertImage(dilatedVentricle,1));
      //calculating pEV and dEV
      pEV=voxelRes * SumImageVoxels(PowerImageToConst(periventricularPmap,k));
      dEV=voxelRes * SumImageVoxels(PowerImageToConst(deepPmap,k));
   }
   catch(itk::ExceptionObject &)
   {
      //if, for example, the 'ventricle' image hasn't been uploaded for some reason, as exception happens.
      std::cerr << "Failed to Quantify WMH segmentations!" << std::endl;
   }

   //saving the outputs of WMH burden calculation into a .txt file.
   try
   {
      std::ofstream EVQuantFile;
      EVQuantFile.open(outputFilename,std::ios::app);
      EVQuantFile << "EV= " << EV << "\n";
      EVQuantFile << "EV-Deep= " << dEV << "\n";
      EVQuantFile << "EV-Periventricular= " << pEV << "\n";
      EVQuantFile.close();
      std::cout << "Quantification of WMH segmentations is saved into: " << outputFilename << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      //if, for example, user doesn't have write permission to the specified file/folder, an exception happens.
      std::cerr << "Failed to save the quantification of WMH segmentations into the file: " << outputFilename << std::endl;
   }  
}//end of QuantifyWMHs()

double GetMinIndexDistanceFromObjects(ImageType2D::IndexType inputIdx,ImagePointer2D objectsBoundaries)
{
   //NOTE: this function calculates the minimum distance of the given index from all boundary voxels of the given 2D boundary image.
   double minDistance;

   itk::Point<int,2> p1;
   p1[0] = inputIdx[0];
   p1[1] = inputIdx[1];

   objectsBoundaries->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionConstIterator<ImageType2D> imageIterator(objectsBoundaries, objectsBoundaries->GetRequestedRegion());
   while(!imageIterator.IsAtEnd())
   {
      itk::Point<int,2> p2;
      p2[0] = imageIterator.GetIndex()[0];
      p2[1] = imageIterator.GetIndex()[1];

      if(imageIterator.IsAtBegin())
         minDistance=p2.EuclideanDistanceTo(p1);
      else if(p2.EuclideanDistanceTo(p1) < minDistance)
         minDistance=p2.EuclideanDistanceTo(p1);
      ++imageIterator;
   }

   return minDistance;
}//end of GetMinIndexDistanceFromBoundaries()

std::vector<ImageType2D::IndexType> FindIndicesByIntensity(ImagePointer2D input,float intensity)
{
   //NOTE: this function returns all indices in the 2D input image, which their intensity value is equal to 'intensity'.
   std::vector<ImageType2D::IndexType> indices;
   input->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionConstIterator<ImageType2D> imageIterator(input, input->GetRequestedRegion());
   while(!imageIterator.IsAtEnd())
   {
      if(imageIterator.Get() == intensity)
         indices.push_back(imageIterator.GetIndex());
      ++imageIterator;
   }
   return indices;
}//end of VoxelCountByIntensity()

ImagePointer2D LabelBinaryObjects(ImagePointer2D input2D, int* objectCount)
{
   /* NOTE:
    * "ConnectedComponentImageFilter labels the objects in a binary image (non-zero pixels are considered to be objects,
    * zero-valued pixels are considered to be background). Each distinct object is assigned a unique label. The filter
    * experiments with some improvements to the existing implementation, and is based on run length encoding along raster lines.
    * The final object labels start with 1 and are consecutive. Objects that are reached earlier by a raster order scan have a lower
    * label. This is different to the behaviour of the original connected component image filter which did not produce consecutive
    * labels or impose any particular ordering."
    */
   typedef itk::Image<unsigned int, 2> OutputImageType2D;   //Note: pixel type of the output in the 'ConnectedComponentImageFilter' filter must be unsigned short/int/...
   typedef OutputImageType2D::Pointer OutputImagePointer2D;
   typedef itk::ConnectedComponentImageFilter<ImageType2D, OutputImageType2D> ConnectedComponentImageFilterType;
   ConnectedComponentImageFilterType::Pointer connectedCompFilter = ConnectedComponentImageFilterType::New();
   connectedCompFilter->SetInput(input2D);
   /**
    * Note: "Face connectivity is 4-connected in 2D, 6 connected in 3D, 2*n in ND.
    * Full connectivity is 8-connected in 2D, 26 connected in 3D, 3^n-1 in ND Default is to use FaceConnectivity."
    */
   connectedCompFilter->SetFullyConnected(true);   // "Set whether the connected components are defined strictly by face connectivity or by face+edge+vertex connectivity. Default is FullyConnectedOff."
   OutputImagePointer2D outputTemp=connectedCompFilter->GetOutput();
   outputTemp->Update();
   *objectCount=connectedCompFilter->GetObjectCount();

   //casting 'outputImageType2D' to 'ImageType2D' to be usable in the rest of the code
   typedef itk::CastImageFilter<OutputImageType2D, ImageType2D> CastFilterType;
   CastFilterType::Pointer castFilter = CastFilterType::New();
   castFilter->SetInput(outputTemp);
   ImagePointer2D output=castFilter->GetOutput();
   output->Update();
   return output;
}//end of LabelBinaryObjects()

ImagePointer ThresholdImage(ImagePointer input, float lowerThreshold, float upperThreshold)
{
   /* Note:
    * "ThresholdImageFilter sets image values to a user-specified "outside" value (by default, "black") if the
    * image values are below, above, or between simple threshold values."
    */
   typedef itk::ThresholdImageFilter<ImageType> ThresholdImageFilterType;
   ThresholdImageFilterType::Pointer thresholdFilter = ThresholdImageFilterType::New();
   thresholdFilter->SetInput(input);
   thresholdFilter->ThresholdOutside(lowerThreshold, upperThreshold);   //other options are: ThresholdAbove() and ThresholdBelow()   Note that "pixels equal to the threshold value are NOT set to OutsideValue in any of these methods"
   thresholdFilter->SetOutsideValue(0);

   ImagePointer output=thresholdFilter->GetOutput();
   output->Update();

   return output;
}//end of ThresholdImage()

ImagePointer2D BinarizeThresholdedImage(ImagePointer2D input2D, float lowerThreshold, float upperThreshold)
{
   /** NOTE:
    * output(xi)=insideValue, if lowerThreshold <= xi <= upperThreshold
    * Otherwise, output(xi)=outsideValue
   **/

   typedef itk::BinaryThresholdImageFilter<ImageType2D, ImageType2D> BinaryThresholdImageFilterType;
   BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
   thresholdFilter->SetInput(input2D);

   thresholdFilter->SetLowerThreshold(lowerThreshold);
   thresholdFilter->SetUpperThreshold(upperThreshold);
   thresholdFilter->SetInsideValue(1);
   thresholdFilter->SetOutsideValue(0);
   ImagePointer2D output=thresholdFilter->GetOutput();
   output->Update();
   return output;
}//end of BinarizeThresholdedImage()

ImagePointer MinMaxNormalisation(ImagePointer input)
{
//   typedef itk::MinimumMaximumImageCalculator<ImageType> ImageCalculatorFilterType;
//   ImageCalculatorFilterType::Pointer imageCalculatorFilter = ImageCalculatorFilterType::New ();
//   imageCalculatorFilter->SetImage(input);
//   imageCalculatorFilter->Compute();
//   ImagePointer output=DivideImageByConstant(input,imageCalculatorFilter->GetMaximum());
//   return output;

   //another way of doing min-max normalisation
   /* "NOTE:
    * In this filter the minimum and maximum values of the input image are computed internally using the
    * MinimumMaximumImageCalculator. Users are not supposed to set those values in this filter. If you need a filter
    * where you can set the minimum and maximum values of the input, please use the IntensityWindowingImageFilter.
    * If you want a filter that can use a user-defined linear transformation for the intensity, then please use the
    * ShiftScaleImageFilter."
    */
   typedef itk::RescaleIntensityImageFilter<ImageType, ImageType> RescaleFilterType;
   RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
   rescaleFilter->SetInput(input);
   rescaleFilter->SetOutputMinimum(0);
   rescaleFilter->SetOutputMaximum(1);   //as the OutputMaximum is set to 1, it behaves like a min-max normaliser...see "http://www.itk.org/Doxygen/html/classitk_1_1RescaleIntensityImageFilter.html"
   ImagePointer output=rescaleFilter->GetOutput();
   output->Update();
   return output;
}//end of MinMaxNormalisation()

ImagePointer InvertImage(ImagePointer input, int maximum)
{
   //Note: InvertIntensityImageFilter inverts intensity of pixels by subtracting pixel value to a maximum value. The maximum value can be set with SetMaximum and defaults the maximum of input pixel type.
   //      This function may also be implemented using the itk::BinaryNotImageFilter when the input image is a binary image
   typedef itk::InvertIntensityImageFilter<ImageType> InvertIntensityImageFilterType;

   InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
   invertIntensityFilter->SetInput(input);
   invertIntensityFilter->SetMaximum(maximum);
   ImagePointer output=invertIntensityFilter->GetOutput();
   output->Update();
   return output;
}//end of InvertImage()

ImagePointer2D Get2DBinaryObjectsBoundaries(ImagePointer2D input)
{
   /*Note: "BinaryContourImageFilter takes a binary image as input, where the pixels in the objects are the pixels
    * with a value equal to ForegroundValue. Only the pixels on the contours of the objects are kept.
    * The pixels not on the border are changed to BackgroundValue."
    */
   typedef itk::BinaryContourImageFilter<ImageType2D, ImageType2D> binaryContourImageFilterType;

   binaryContourImageFilterType::Pointer binaryContourFilter = binaryContourImageFilterType::New();
   binaryContourFilter->SetInput(input);
   binaryContourFilter->SetBackgroundValue(0);      //setting background and foreground values are necessary.
   binaryContourFilter->SetForegroundValue(1);
   ImagePointer2D output=binaryContourFilter->GetOutput();
   output->Update();
   return output;

}//end of Get2DBinaryObjectsBoundaries()

ImagePointer2D Get2DSlice(ImagePointer input3D,int plane, int slice)
{
   /* Note:
    * 'slice': represents the slice number to be extracted (along the specified dimension/plane)
    * 'plane': for example, if we would like to get slices of an image along the Z-axis, we set the 'plane' to 2.
    */
   typedef itk::ExtractImageFilter<ImageType, ImageType2D> extractImageFilterType;
   extractImageFilterType::Pointer extractImageFilter = extractImageFilterType::New();

   ImageType::RegionType inputRegion = input3D->GetLargestPossibleRegion();

   ImageType::SizeType size = inputRegion.GetSize();
   size[plane] = 0;   //'plane' is the dimension number on which we would like to extract a slice of the image.

   ImageType::IndexType start = inputRegion.GetIndex();
   start[plane] = slice;

   ImageType::RegionType desiredRegion;
   desiredRegion.SetSize(size);
   desiredRegion.SetIndex(start);

   extractImageFilter->SetDirectionCollapseToSubmatrix();   //"Set the strategy to be used to collapse physical space dimensions."
   extractImageFilter->SetExtractionRegion(desiredRegion);

   extractImageFilter->SetInput(input3D);

   ImagePointer2D output2D = extractImageFilter->GetOutput();
   output2D->Update();

   return output2D;
}//end of Get2DSlice()

bool LoadTrainingDataset(std::string trainingFilename, Mat trainingFeatures, Mat trainingLabels,int samplesCount,int featuresCount)
{
   //Note: the dataset should be a .csv file, which contains both features and labels
   //NOTE: the following method of reading the training data set is about 4 times faster than using 'fopen()', 'fscan()', etc. for a data set of about 4000 samples.
   std::cout << "Loading training dataset..." << std::endl;

   CvMLData trainingData;
   if(trainingData.read_csv(trainingFilename.c_str()) == 0)
   {

   //   trainingData.change_var_type(featuresCount,CV_VAR_ORDERED);
      trainingData.set_response_idx(featuresCount);

      Mat tempLabels=trainingData.get_responses();
      tempLabels.copyTo(trainingLabels);

      Mat tempFeatures=trainingData.get_values();
      tempFeatures=tempFeatures.colRange(0,featuresCount);   //NOTE: "'startcol': An inclusive 0-based start index of the column span. 'endcol': An exclusive 0-based ending index of the column span."
      for(int sIdx=0; sIdx < tempFeatures.size[0]; ++sIdx)
         for(int fIdx=0; fIdx < tempFeatures.size[1]; ++fIdx)
            trainingFeatures.at<float>(sIdx,fIdx)=tempFeatures.at<float>(sIdx,fIdx);

      std::cout << "Training data loaded successfully!" << std::endl;
      return true;
   }
   else
   {
      std::cerr << "Error in loading training data: " << trainingFilename << std::endl;
      return false;
   }
}//end of LoadTrainingDataset()

bool LoadTestingDataset(std::string testingFilename, Mat testingFeatures, Mat testingLabels,int samplesCount,int featuresCount)
{   //Note: the dataset should be a .csv file, which contains both features and labels
   std::cout << "Loading testing dataset..." << std::endl;

    FILE* file = fopen(testingFilename.c_str(), "r");
    if(!file)
    {
        std::cerr << "Cannot read file: " <<  testingFilename << std::endl;
        return false;
    }
    else
    {
       float feature;
      for(int sampleIdx = 0; sampleIdx < samplesCount; ++sampleIdx)
      {
         for(int featureIdx = 0; featureIdx < (featuresCount+1); ++featureIdx)   //FOR cross-validation
//         for(int featureIdx = 0; featureIdx < featuresCount; ++featureIdx)
         {
            if (featureIdx < featuresCount)
            {
               fscanf(file, "%f,", &feature);
               testingFeatures.at<float>(sampleIdx, featureIdx) = feature;
            }
            else if (featureIdx == featuresCount)
            {
               fscanf(file, "%f,", &feature);//just to pass the testing set label
               testingLabels.at<float>(sampleIdx,0)=feature;
//               std::cout << "actual:\t" << feature << std::endl;
            }
         }//end of for feature loop
      }//end of for sample loop

      fclose(file);

      std::cout << "Testing data is loaded successfully!" << std::endl;
      return true;
    }
}//end of LoadTrainingDataset()


CvRTrees* GenOrLoadRFModel(std::string modelFilename)
{
   CvRTrees* rfModel=new CvRTrees();
   
      std::cout << "Loading RF model: " << modelFilename << "..." << std::endl;
      rfModel->load(modelFilename.c_str(),"envisionRFModel");      //loads the pre-generated and saved RF model
   

   std::cout << "RF model (" << modelFilename << ") generated/loaded successfully!" << std::endl;
   return rfModel;
}// end of GenOrLoadRFModel()


CvRTrees* GenOrLoadRFModel(std::string modelFilename, Mat trainingFeatures, Mat trainingLabels)
{
   CvRTrees* rfModel=new CvRTrees();
   
      std::cout << "Generating RF model..." << std::endl;

      Mat var_type = Mat(trainingFeatures.size[1] + 1, 1, CV_8U);         //CV_8UC1 means a 8-bit single-channel array
      var_type.setTo(Scalar(CV_VAR_NUMERICAL));                      //all elements are numerical, except for the last one (see bellow)
      var_type.at<char>(trainingFeatures.size[1], 0) = CV_VAR_ORDERED;   //set the last element of var_type element to CV_VAR_CATEGORICAL
                                                         //for regression: set to CV_VAR_ORDERED
      /* CvRTParams constructor parameters are as follows:
       * 1.max depth of trees,
       * 2.min sample count ('minparent' in Matlab's stochastic_bosque
       * 3.regression accuracy,
       * 4.compute surrogate split (no missing data),
       * 5.max number of categories (use sub-optimal algorithm for larger numbers),
       * 6.the array of weights of each classification for classes,
       * 7.calculate variable importance,
       * 8.number of variables randomly selected at node and used to find the best split(s). If 0, it'll get the square root of the total number of features per sample.
       * 9.max number of trees in the forest ('ntrees' in Matlab's stochastic_bosque()),
       * 10.forest accuracy (OOB error)
       * 11.termination criteria:      CV_TERMCRIT_ITER: terminate after the maximum number of iterations or elements to compute is reached.
                               CV_TERMCRIT_EPS: terminate after the desired accuracy or change in parameters (at which the iterative algorithm stops) is reached.
                               CV_TERMCRIT_ITER | CV_TERMCRIT_EPS: terminate either after the first or the second condition is satisfied.
       */

      float priors[] = {1,1};  // weights of each class for classification
      CvRTParams params = CvRTParams(6, 4, 0.0f, false, 2, 0, false, 90, 50, 0.01f, CV_TERMCRIT_ITER);

         /* 'tflag': a flag showing how do samples stored in the trainData matrix:
          *  1. row by row (tflag=CV_ROW_SAMPLE) or
          *  2. column by column (tflag=CV_COL_SAMPLE).
          */
         rfModel->train(trainingFeatures, CV_ROW_SAMPLE, trainingLabels, Mat(), Mat(), var_type, Mat(), params);

         //saving the Random Forest regressionmodel into an .xml file.
         rfModel->save(modelFilename.c_str(),"envisionRFModel");
   
   
   std::cout << "RF model (" << modelFilename << ") generated/loaded successfully!" << std::endl;
   return rfModel;
}// end of GenOrLoadRFModel()




ImagePointer NiftiReader(char * inputFile)
{
   typedef itk::ImageFileReader<ImageType> ImageReaderType;
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();
   ImageReaderType::Pointer imageReader=ImageReaderType::New();
   ImagePointer output;
   try
   {
      niftiIO->SetFileName(inputFile);
      niftiIO->ReadImageInformation();

      imageReader->SetImageIO(niftiIO);
      imageReader->SetFileName(niftiIO->GetFileName());
      output=imageReader->GetOutput();
      output->Update();
      std::cout << "Successfully read: " << inputFile << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      std::cerr << "Failed to read: " << inputFile << std::endl;
   }
   return output;
}//end of NiftiReader()

ImagePointer NiftiReader(std::string inputFile)
{
   typedef itk::ImageFileReader<ImageType> ImageReaderType;
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();
   ImageReaderType::Pointer imageReader=ImageReaderType::New();
   ImagePointer output;
   try
   {
      niftiIO->SetFileName(inputFile);
      niftiIO->ReadImageInformation();

      imageReader->SetImageIO(niftiIO);
      imageReader->SetFileName(niftiIO->GetFileName());
      output=imageReader->GetOutput();
      output->Update();
      std::cout << "Successfully read: " << inputFile << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      std::cerr << "Failed to read: " << inputFile << std::endl;
   }
   return output;
}//end of NiftiReader()

bool NiftiWriter(ImagePointer input,std::string outputFile)
{
   typedef itk::ImageFileWriter<ImageType> imageWriterType;
   imageWriterType::Pointer imageWriterPointer =imageWriterType::New();
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();

   try
   {
      //Set the output filename
      imageWriterPointer->SetFileName(outputFile);
      //Set input image to the writer.
      imageWriterPointer->SetInput(input);
      //Determine file type and instantiate appropriate ImageIO class if not explicitly stated with SetImageIO, then write to disk.
      imageWriterPointer->SetImageIO(niftiIO);

      imageWriterPointer->Write();
      std::cout << "Successfully saved image: " << outputFile << std::endl;
      return true;
   }
   catch ( itk::ExceptionObject & ex )
   {
      std::string message;
      message = "Problem found while saving image ";
      message += outputFile;
      message += "\n";
      message += ex.GetLocation();
      message += "\n";
      message += ex.GetDescription();
      std::cerr << message << std::endl;
      return false;
   }
}//end of NiftiWriter()

ImagePointer GetPatch(ImagePointer input,ImageType::IndexType centreIdx,unsigned int ROIsize)
{
     typedef itk::RegionOfInterestImageFilter<ImageType, ImageType> ROIFilterType;
     ROIFilterType::Pointer ROIfilter = ROIFilterType::New();

     ImageType::IndexType startIdx;
     startIdx[0]=centreIdx[0]-(ROIsize/2);
     startIdx[1]=centreIdx[1]-(ROIsize/2);
     startIdx[2]=centreIdx[2]-(ROIsize/2);

     ImageType::SizeType size;
     size.Fill(ROIsize);   //Fill() sets one value for the index in all dimensions.

     ImageType::RegionType desiredRegion;
     desiredRegion.SetSize(size);
     desiredRegion.SetIndex(startIdx);

     ImagePointer outputROI;

     try
     {
        ROIfilter->SetRegionOfInterest(desiredRegion);
        ROIfilter->SetInput(input);
        outputROI=ROIfilter->GetOutput();
        outputROI->Update();

        //std::cout << "ROI extracted successfully!" << std::endl;
     }
     catch(itk::ExceptionObject & ex)
     {
        std::cerr << "A problem encountered when extracting a patch of 5*5*5 voxels!" << std::endl;
//        std::cerr << ex << std::endl;
     }
     return outputROI;
}//end of GetPatch()

std::list<ImagePointer> GetAllKernels()
{
   std::list<ImagePointer> kernels;
   kernels.push_back(CreateGaussianKernel(-1,3));
   kernels.push_back(CreateLaplacianOfGKernel(-1,5));

   std::list<ImagePointer> DoGs=CreateDifferenceOfGLKernels(true,(std::array<int,2>){3,5},(std::array<double,4>){0.5,1.1,1.75,2.5});
   kernels.splice(kernels.end(),DoGs);      //void splice (const_iterator position, list& x); //transfers all the elements of x into the container.

   std::list<ImagePointer> DoLs=CreateDifferenceOfGLKernels(false,(std::array<int,2>){3,5},(std::array<double,4>){0.75,0.95,1.2,1.4});
   kernels.splice(kernels.end(),DoLs);

   kernels.push_back(CreateSobelKernel());

   return kernels;
}//end of GetAllKernels()

ImagePointer CreateGaussianKernel(float variance,int width)
{
   /* DONE: 'ImagePointer GaussianImageFilter(ImagePointer inputImage,double variance);' was implemented to
   * compared the efficiency of the itk::DiscreteGaussianImageFilter with the kernels introduced
   * in the W2MHS toolbox. Use of those kernels requires using ConvolutionImageFilter class, which probably
   * doesn't support multithreading, as opposed to DiscreteGaussianImageFilter.
   * PERFORMANCE TEST1 RESULT: for larger patches, the DiscreteGaussianImageFilter is quicker than KernelConvolotion.
   *
   *    PERFORMANCE TEST2 RESULT: 1000000 patches of size 5*5*5 were tested with both DiscreteGaussianImageFilter
   *                        and Kernel Convolution.
   *                        The Kernel Convolution was twice faster than DiscreteGaussianImageFilter.
   *
   */

   if(variance <= 0)
      variance=2.5*std::sqrt(width);      //according to the W2MHS toolbox

   int radius=width/2;
   ImagePointer kernelImage=ImageType::New();

   ImageType::IndexType startIdx;
   startIdx.Fill(0);

   ImageType::SizeType kernelSize;
   kernelSize.Fill(width);

   ImageType::RegionType region;
   region.SetSize(kernelSize);
   region.SetIndex(startIdx);

   kernelImage->SetRegions(region);
   kernelImage->Allocate();      //Allocate the image memory. The size of the image must already be set, e.g. by calling SetRegions().

   itk::ImageRegionIterator<ImageType> imageIterator(kernelImage, region);      //kernel's region and region are equal here
   for(int k=-radius;k<=radius;++k)
      for(int j=-radius;j<=radius;++j)
         for(int i=-radius;i<=radius;++i)
            if(!imageIterator.IsAtEnd())
            {
               float r=i*i+j*j+k*k;
               imageIterator.Set(std::exp(-r/(2*variance*variance)));
               ++imageIterator;
            }
   return kernelImage;
}//end of CreateGaussianKernel()

ImagePointer CreateLaplacianOfGKernel(float variance,int width)
{
   if(variance <= 0)
      variance=1.095*std::sqrt(width);      //according to the W2MHS toolbox

   int radius=width/2;
   double kernelMin,kernelMax;
   ImagePointer kernelImage=ImageType::New();

   ImageType::IndexType startIdx;
   startIdx.Fill(0);

   ImageType::SizeType kernelSize;
   kernelSize.Fill(width);

   ImageType::RegionType region;
   region.SetSize(kernelSize);
   region.SetIndex(startIdx);

   kernelImage->SetRegions(region);
   kernelImage->Allocate();      //Allocate the image memory. The size of the image must already be set, e.g. by calling SetRegions().

   itk::ImageRegionIterator<ImageType> imageIterator(kernelImage, region);      //kernel's region and region are equal here
   for(int k=-1*radius;k<=radius;++k)
      for(int j=-1*radius;j<=radius;++j)
         for(int i=-1*radius;i<=radius;++i)
            if(!imageIterator.IsAtEnd())
            {
               float r=i*i+j*j+k*k;
               imageIterator.Set(((r-(3*variance*variance))/std::pow(variance,4))*std::exp(-r/(2*variance*variance)));
               if(imageIterator.IsAtBegin())
               {   //the initial values of the min and max are set in the first loop
                  kernelMin=imageIterator.Get();
                  kernelMax=imageIterator.Get();
               }
               else
               {
                  if(imageIterator.Get() > kernelMax)
                     kernelMax=imageIterator.Get();
                  else if(imageIterator.Get() < kernelMin)
                     kernelMin=imageIterator.Get();
               }
               ++imageIterator;
            }

   //according to the W2MHS toolbox, the absolute value of the kernel minimum and maximum should be used for the final scaling on the kernel
   kernelMin=std::abs(kernelMin);
   kernelMax=std::abs(kernelMax);

   imageIterator.GoToBegin();   //the imageIterator should go to the beginning of the kernel image so that the scaling operation, as performed in the W2MHS toolbox, can be done.
   while(!imageIterator.IsAtEnd())
   {
      imageIterator.Set((imageIterator.Get()+((kernelMin-kernelMax)/2))*(2/(kernelMax+kernelMin)));
      ++imageIterator;
   }
   return kernelImage;
}//end of CreateLaplacianOfGKernel()

std::list<ImagePointer> CreateDifferenceOfGLKernels(bool GaussianOrLaplacian,std::array<int,2> width,std::array<double,4> baseVarianceArray)
{
   /* Note:
    * If 'GaussianOrLaplacian'=true, this method creates DifferenceOfGaussians kernels.
    * Otherwise, it creates DifferenceOfLaplacians kernels.
    *
    * The values of 'width' and 'baseVarianceArray' arrays are set according to the W2MHS toolbox.
    */

   /*NOTE: This method can be implemented more efficiently (without re-using the CreateGaussianKernel()/CreateLaplacianKernel() methods), if needed.
    *      Currently, it runs in about 1 millisecond. So, there shouldn't be much difference if it's implemented without re-using the CreateGaussianKernel()/CreateLaplacianKernel() methods.
    *      As a result, it's been decided to re-use the CreateGaussianKernel()/CreateLaplacianKernel() methods here.
    */
   std::list<ImagePointer> DoGLKernels;

   for(short w=0;w<width.size();++w)
   {
      for(int i=1;i<baseVarianceArray.size();++i)
      {
         //calculating the difference of two Gaussian/Laplacian Kernels
         typedef itk::SubtractImageFilter<ImageType,ImageType> SubtractImageFilterType;
         SubtractImageFilterType::Pointer subtractFilter=SubtractImageFilterType::New ();
         if(GaussianOrLaplacian)
         {//If 'GaussianOrLaplacian'=true, this method creates 6 DifferenceOfGaussians kernels.
            subtractFilter->SetInput1(CreateGaussianKernel(baseVarianceArray[i]*std::sqrt(width[w]),width[w]));
            subtractFilter->SetInput2(CreateGaussianKernel(baseVarianceArray[i-1]*std::sqrt(width[w]),width[w]));
         }
         else
         {//If 'GaussianOrLaplacian'=false, this method creates 6 DifferenceOfLaplacians kernels.
            subtractFilter->SetInput1(CreateLaplacianOfGKernel(baseVarianceArray[i]*std::sqrt(width[w]),width[w]));
            subtractFilter->SetInput2(CreateLaplacianOfGKernel(baseVarianceArray[i-1]*std::sqrt(width[w]),width[w]));
         }
         subtractFilter->Update();
         //
         DoGLKernels.push_back(subtractFilter->GetOutput());   //adds the new DoG kernel at the end of the list

      }
   }

   return DoGLKernels;
}//end of CreateDifferenceOfGLKernels()



ImagePointer CreateSobelKernel()
{
   /*TODO: The W2MHS sobel filter creates something very similar to the Gaussian filter output.
    *   So, basically it doesn't add any more information to the feature vector.
    *   However, using ITK library we can create a sobel filter that actually detects edges (itk::SobelEdgeDetectionImageFilter).
    *   So, we might be able to create more informative/useful feature vectors.
   */

   int width=3;      //Sobel kernel's width=3
   short h[]={1,2,1};
   short h1[]={-1,0,1};
   int sobelX[3][3][3];
   int sobelY[3][3][3];
   int sobelZ[3][3][3];
   for(int k=0;k<width;++k)
      for(int j=0;j<width;++j)
      {
         for(int i=0;i<width;++i)
         {
            sobelX[i][k][j]=h[j]*h[i]*h1[k];
            sobelY[k][i][j]=h[j]*h[i]*h1[k];
            sobelZ[i][j][k]=h[j]*h[i]*h1[k];
         }
      }

   ImagePointer kernelImage=ImageType::New();

   ImageType::IndexType startIdx;
   startIdx.Fill(0);

   ImageType::SizeType kernelSize;
   kernelSize.Fill(width);

   ImageType::RegionType region;
   region.SetSize(kernelSize);
   region.SetIndex(startIdx);

   kernelImage->SetRegions(region);
   kernelImage->Allocate();      //Allocate the image memory. The size of the image must already be set, e.g. by calling SetRegions().

   itk::ImageRegionIterator<ImageType> imageIterator(kernelImage, region);   //kernel's region and region are equal here

   for(int k=0;k<width;++k)
         for(int j=0;j<width;++j)
            for(int i=0;i<width;++i)
            {
               if(!imageIterator.IsAtEnd())
               {
                  //calculating the magnitude of the gradient vector
                  imageIterator.Set(std::sqrt((sobelX[i][j][k]*sobelX[i][j][k])+(sobelY[i][j][k]*sobelY[i][j][k])+(sobelZ[i][j][k]*sobelZ[i][j][k])));
                  ++imageIterator;
               }
            }

   return kernelImage;
}//end of CreateSobelKernel()

ImagePointer ConvolveImage(ImagePointer input,ImagePointer kernel,bool normaliseOutput)
{
   /* Note:
    * in order for the result of this convolution method to be matched with ones from the convn() method
    * in Matlab, we should first surround the input image (i.e. 'patch' in this case) with zeros (the size of the zero margin should be equal to the input image size + twice the floor of half the kernel size --->e.g. 5+(2*(3/1))=5+(2*1)=7).
    * This would result in the same behaviour as convn(in,ker,'same').
    */
   unsigned int inputImgSize=input->GetLargestPossibleRegion().GetSize()[0];
   unsigned int kernelSize=kernel->GetLargestPossibleRegion().GetSize()[0];
   unsigned int szImageSize=inputImgSize+(2*std::floor(kernelSize/2));

   ImagePointer zeroSurroundedImage=ImageType::New();

   ImageType::IndexType zsStartIdx;
   zsStartIdx.Fill(0);

   ImageType::SizeType zsSize;
   zsSize.Fill(szImageSize);

   ImageType::RegionType zsRegion;
   zsRegion.SetSize(zsSize);
   zsRegion.SetIndex(zsStartIdx);

   zeroSurroundedImage->SetRegions(zsRegion);
   zeroSurroundedImage->SetDirection(input->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
   zeroSurroundedImage->SetSpacing(input->GetSpacing());      //e.g. 2mm*2mm*2mm
   zeroSurroundedImage->SetOrigin(input->GetOrigin());
   zeroSurroundedImage->Allocate();

   itk::ImageRegionIterator<ImageType> zsIterator(zeroSurroundedImage, zsRegion);
   itk::ImageRegionIterator<ImageType> imgIterator(input, input->GetLargestPossibleRegion());

   for(int i=0;i<szImageSize;++i)
      for(int j=0;j<szImageSize;++j)
         for(int k=0;k<szImageSize;++k)
         {
            if(!zsIterator.IsAtEnd())
            {
               //a margin of zeros with half the kernel size should surround the input image
               if((i>=0 && i<std::floor(kernelSize/2)) || (i>szImageSize-1-std::floor(kernelSize/2) && i<=szImageSize-1)
                     || (j>=0 && j<std::floor(kernelSize/2)) || (j>szImageSize-1-std::floor(kernelSize/2) && j<=szImageSize-1)
                     || (k>=0 && k<std::floor(kernelSize/2)) || (k>szImageSize-1-std::floor(kernelSize/2) && k<=szImageSize-1))
                  zsIterator.Set(0);
               else
               {
                  zsIterator.Set(imgIterator.Get());
                  ++imgIterator;
               }
               ++zsIterator;
            }
         }

   //convolving the zero surrounded input image with the kernel
   //TODO: itk::FFTConvolutionImageFilter can be used for more efficiency
   typedef itk::ConvolutionImageFilter<ImageType> ConvolutionType;
   ConvolutionType::Pointer convolution = ConvolutionType::New();
   ImagePointer output;
   try
   {
      convolution->SetInput(zeroSurroundedImage);
      #if ITK_VERSION_MAJOR >= 4
         convolution->SetKernelImage(kernel);
      #else
         convolutionFilter->SetImageKernelInput(kernel);
      #endif
      if(normaliseOutput)
         convolution->NormalizeOn();      //Normalize the output image by the sum of the kernel components. Defaults to off.
      output=convolution->GetOutput();
      output->Update();
      //std::cout << "Convolution successful!" << std::endl;
   }
   catch(itk::ExceptionObject & ex)
   {
      std::cerr << "A problem occured when performing image convolution!" << std::endl;
//      std::cerr << ex << std::endl;
   }

   //get the central part of the output with the same size as the input (as the convn() method in Matlab works)
   ImageType::IndexType outStartIdx;
   outStartIdx.Fill(std::floor(kernelSize/2));      //the convn() method in Matlab: For the 'same' case, conv2 returns the central part of the convolution. If there are an odd number of rows or columns, the "center" leaves one more at the beginning than the end.
   ImageType::SizeType outSize;
   outSize.Fill(inputImgSize);
   ImageType::RegionType outRegion;
   outRegion.SetSize(outSize);
   outRegion.SetIndex(outStartIdx);
   output->SetLargestPossibleRegion(outRegion);

   return output;
}


ImagePointer ConvolveImage2(ImagePointer  patch,ImagePointer kernel)
{
   ConvolutionType::Pointer convolution = ConvolutionType::New();
    ImagePointer output;
    try
    {
      convolution->SetInput(patch);
      #if ITK_VERSION_MAJOR >= 4
         convolution->SetKernelImage(kernel);
      #else
         convolutionFilter->SetImageKernelInput(kernel);
      #endif
           
      output=convolution->GetOutput();
      output->Update();
      
    }
    catch(itk::ExceptionObject & ex)
    {
       std::cerr << "A problem occured when performing image convolution!" << std::endl;

    }
    return output;
}

void MarginateImage(ImagePointer input,int marginWidth)
{
   //Note: itk::CropImageFilter hasn't been used for this purpose, as it throws away the cropped margin (which is not what we need here).
   //this function receives an Image Pointer. So, the changes that are applied to the image that exists at the specified pointer will remain (even) after returning from this function.
   //this function is used in W2MHS toolbox (probably) to avoid falling outside of the image boundaries while extracting patches for each voxel.
   //If this is the case, there should also be no problem if we marginate the image by 1/2 of the patch size. This is because each patch around each voxel has a "radius" equal to 1/2 of the patch size.
   //TODO: (issue120302) it may not really be necessary to set this margin to 0. Instead, we can probably create a region with the size of this margin
   //      and then create an imageIterator which only iterates inside this margin of the input image ---> itk::ImageRegionIterator<ImageType> imageIterator(input, marginRegion);
   input->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> imageIterator(input, input->GetRequestedRegion());   //for iteration through the whole image
   while(!imageIterator.IsAtEnd())
   {
      ImageType::IndexType currentIdx=imageIterator.GetIndex();
      if(currentIdx[0] < marginWidth || currentIdx[1] < marginWidth || currentIdx[2] < marginWidth
            || currentIdx[0] > (input->GetRequestedRegion().GetSize()[0]-1-marginWidth)
            || currentIdx[1] > (input->GetRequestedRegion().GetSize()[1]-1-marginWidth)
            || currentIdx[2] > (input->GetRequestedRegion().GetSize()[2]-1-marginWidth))
         imageIterator.Set(0);
      ++imageIterator;
   }
}//end of MarginateImage()

double GetHistogramMax(ImagePointer inputImage,unsigned int binsCount)
{
   const unsigned int MeasurementVectorSize = 1; //Grayscale
   typedef itk::Statistics::ImageToHistogramFilter<ImageType> ImageToHistogramFilterType;
   typedef ImageToHistogramFilterType::HistogramType HistogramType;
   HistogramType::SizeType size(MeasurementVectorSize);
   size.Fill(binsCount);

   ImageToHistogramFilterType::Pointer imageToHistogramFilter = ImageToHistogramFilterType::New();
   imageToHistogramFilter->SetInput(inputImage);
   imageToHistogramFilter->SetHistogramSize(size);
   imageToHistogramFilter->Update();

   HistogramType* histogram = imageToHistogramFilter->GetOutput();
   int histMaxIdx=0,histMax=0;
   for(unsigned int i = 1; i < histogram->GetSize()[0]; ++i)
   {
      if(histogram->GetFrequency(i) > histMax)
      {
        histMax=histogram->GetFrequency(i);
        histMaxIdx=i;
      }
   }

   return histogram->GetBinMax(0,histMaxIdx);   //Get the maximum value of nth bin of dimension d
}//end of GetHistogramMax()

void WriteImageFeatureVectorToFile(ImagePointer input,bool newPatch,std::string outputFilename)
{
   //NOTE: THIS METHOD IS FOR TESTING AND DEBUGGING

   std::ofstream featuresFile;
   featuresFile.open(outputFilename,std::ios::app);   //std::ios::app is for the file to be opened in the 'appending' mode

   if(newPatch)
      featuresFile << "\n";
   input->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> imageIterator(input, input->GetRequestedRegion());
   while(!imageIterator.IsAtEnd())
   {
      featuresFile << imageIterator.Get() << ",";
      ++imageIterator;
   }

   featuresFile.close();
}//end of WriteImageFeatureVectorToFile()

void AppendToPatchFeatureVector(ImagePointer patch, Mat patchFeatureMat, int startIdx)
{
   patch->SetRequestedRegionToLargestPossibleRegion();
   
   //itk::ImageRegionIterator<ImageType> patchIterator(patch, patch->GetRequestedRegion());
   //itk::ImageLinearIteratorWithIndex<ImageType> patchIterator(patch, patch->GetRequestedRegion());
   itk::ImageRegionIteratorWithIndex<ImageType> patchIterator(patch, patch->GetRequestedRegion());
   
   while(!patchIterator.IsAtEnd())
   {
      patchFeatureMat.at<float>(0,startIdx)=patchIterator.Get();
      ++patchIterator;
      ++startIdx;
   }
}//end of AppendToPatchFeatureVector()

void CreatePatchFeatureVector(ImagePointer patch, Mat patchFeatureMat)
{
   //get all kernels as a list. Kernels should be calculated first and be used as many times as needed.
   std::list<ImagePointer> kernels=GetAllKernels();
   
   int startIdx=0;      //this is to tell the AppendToPatchFeatureVector(), where to add new features
   //get voxel intensities of each voxel of the path and add them to the feature vector of the patch
   AppendToPatchFeatureVector(patch,patchFeatureMat,startIdx);

   for (std::list<ImagePointer>::iterator kernelsIterator=kernels.begin(); kernelsIterator!=kernels.end(); ++kernelsIterator)
   {
      startIdx+=(patch->GetLargestPossibleRegion().GetSize()[0]*patch->GetLargestPossibleRegion().GetSize()[1]*patch->GetLargestPossibleRegion().GetSize()[2]);
      //1. convolve the patch with all the kernels one by one
      //2. get values of the returned convolved image and add them to the feature vector of the patch
      ImagePointer convolvedImg=ConvolveImage(patch,*kernelsIterator,false);
      //NOTE: some scaling has been done on the convolved image in the W2MHS toolbox, which are done here as well.
      //      When we can get our own ground truth data, we may not need to do these necessarily.
         int kernelIdx=distance(kernels.begin(), kernelsIterator);
         int scalediv=1;
         if(kernelIdx == kernels.size()-1)
            scalediv=3;
         int kernelWidth=kernelsIterator->GetPointer()->GetLargestPossibleRegion().GetSize()[0];
         convolvedImg=DivideImageByConstant(convolvedImg,scalediv*std::pow(kernelWidth,3));

         if(kernelIdx >= 2 && kernelIdx <= 7)
            convolvedImg=AddImageToImage(convolvedImg,patch,false);
         else if(kernelIdx >= 8 && kernelIdx <= 13)
            convolvedImg=AddImageToImage(convolvedImg,patch,true);
      //
      AppendToPatchFeatureVector(convolvedImg,patchFeatureMat,startIdx);
   }
}//end of CreatePatchFeatureVector()


ImagePointer MultiplyTwoImages(ImagePointer input1,ImagePointer input2)
{
   //Note: Pixel-wise multiplication of two images

   typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyImageFilterType;
   MultiplyImageFilterType::Pointer multiplyImageFilter = MultiplyImageFilterType::New();
   multiplyImageFilter->SetInput1(input1);
   multiplyImageFilter->SetInput2(input2);
   ImagePointer output=multiplyImageFilter->GetOutput();
   output->Update();
   return output;
}//MultiplyTwoImages()

ImagePointer DivideImageByConstant(ImagePointer input,double constant)
{
   //NOTE: itk::MultiplyByConstantImageFilter is deprecated

//   typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyImageFilterType;
//   MultiplyImageFilterType::Pointer multiplyImageFilter = MultiplyImageFilterType::New();
//   multiplyImageFilter->SetInput1(input);
//   multiplyImageFilter->SetConstant2(1/constant);
//   ImagePointer output=multiplyImageFilter->GetOutput();
//   output->Update();
//   return output;


   //NOTE: the following implementation would run about 2.5 times faster than MultiplyImageFilter

   ImagePointer outImage=ImageType::New();

   ImageType::IndexType outStartIdx;
   outStartIdx.Fill(0);

   ImageType::SizeType outSize=input->GetLargestPossibleRegion().GetSize();

   ImageType::RegionType outRegion;
   outRegion.SetSize(outSize);
   outRegion.SetIndex(outStartIdx);

   outImage->SetRegions(outRegion);
   outImage->SetDirection(input->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
   outImage->SetSpacing(input->GetSpacing());   //e.g. 2mm*2mm*2mm
   outImage->SetOrigin(input->GetOrigin());
   outImage->Allocate();

   itk::ImageRegionIterator<ImageType> outIterator(outImage, outRegion);
   itk::ImageRegionIterator<ImageType> inpIterator(input, input->GetLargestPossibleRegion());
   while(!outIterator.IsAtEnd())
   {
      outIterator.Set(inpIterator.Get() / constant);

      ++outIterator;
      ++inpIterator;
   }
   return outImage;

}//end of DivideImageByConstant()

ImagePointer AddImageToImage(ImagePointer input1,ImagePointer input2, bool isSubraction)
{
   //NOTE: the following implementation would run about twice faster than SubtractImageFilter/AddImageFilter

   ImagePointer outImage=ImageType::New();

   ImageType::IndexType outStartIdx;
   outStartIdx.Fill(0);

   ImageType::SizeType outSize=input1->GetLargestPossibleRegion().GetSize();

   ImageType::RegionType outRegion;
   outRegion.SetSize(outSize);
   outRegion.SetIndex(outStartIdx);

   outImage->SetRegions(outRegion);
   outImage->SetDirection(input1->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
   outImage->SetSpacing(input1->GetSpacing());   //e.g. 2mm*2mm*2mm
   outImage->SetOrigin(input1->GetOrigin());
   outImage->Allocate();

   itk::ImageRegionIterator<ImageType> outIterator(outImage, outRegion);
   itk::ImageRegionIterator<ImageType> inp1Iterator(input1, input1->GetLargestPossibleRegion());
   itk::ImageRegionIterator<ImageType> inp2Iterator(input2, input2->GetLargestPossibleRegion());
   while(!outIterator.IsAtEnd())
   {
      if(!isSubraction)
         outIterator.Set(inp1Iterator.Get() + inp2Iterator.Get());
      else
         outIterator.Set(inp1Iterator.Get() - inp2Iterator.Get());
      ++outIterator;
      ++inp1Iterator;
      ++inp2Iterator;
   }
   return outImage;
}//end of AddImageToImage()

float SumImageVoxels(ImagePointer input)
{
   /* Note: "StatisticsImageFilter computes the minimum, maximum, sum, mean, variance sigma of an image."
    * Thus, another input parameter can be passed to this function, to indicate which type of statistics is required as the output.
    *
    * For example, 'statOutputNO' (which would be the second input parameter of the function) options may be considered as follows:
    *       sum: 1
    *       minimum: 2
    *       maximum: 3
    *       mean: 4
    *       variance: 5
    *
    *    Then, we can output the required statistics of the image as follows:
    *    float output=0;
    *      switch(statNO)
    *      {
    *         case 1:
    *            output=statisticsImageFilter->GetSum();
    *            break;
    *         case 2:
    *            output=statisticsImageFilter->GetMinimum();
    *            break;
    *         case 3:
    *            output=statisticsImageFilter->GetMaximum();
    *            break;
    *         case 4:
    *            output=statisticsImageFilter->GetMean();
    *            break;
    *         case 5:
    *            output=statisticsImageFilter->GetVariance();
    *            break;
    *         default:
    *            break;
    *      }
    *      return output;
    */

   typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
   StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
   statisticsImageFilter->SetInput(input);
   statisticsImageFilter->Update();

   return statisticsImageFilter->GetSum();
}//end of SumImageVoxels()

ImagePointer PowerImageToConst(ImagePointer input, float constPower)
{
   if(constPower == 1 )
      return input;
   else
   {
      typedef itk::PowImageFilter<ImageType,ImageType,ImageType> PowImageFilterType;
      PowImageFilterType::Pointer powImageFilter = PowImageFilterType::New();
      powImageFilter->SetInput1(input);
      powImageFilter->SetConstant2(constPower);   //Note: "Additionally, this filter can be used to raise every pixel of an image to a power of a constant by using SetConstant2()."
      ImagePointer output=powImageFilter->GetOutput();
      output->Update();

      return output;
   }
}//end of PowerImageToConst()

//void PerformanceTest(std::string message)
//{
//   milliseconds msend = duration_cast< milliseconds >(high_resolution_clock::now().time_since_epoch());
//   std::cout << message << msend.count() << std::endl;
//}

void CalculateMSE(ImagePointer img1,ImagePointer img2)
{
   //Note: This function is for TEST & DEBUG
   /*This function is to calculate the MSE of the testing data set, when the 'labels' are not saved in the testing data set as the last column.
    *The final prediction values, which are saved into a 'RFSegOut...nii' image, will be checked against the actual segmentation values, which are saved into a 'RFREG_out_...nii' file.
    */
   float MSE=0;
   int voxelCount=0;
   img1->SetRequestedRegionToLargestPossibleRegion();
   img2->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionConstIterator<ImageType> imgIterator1(img1, img1->GetRequestedRegion());
   itk::ImageRegionConstIterator<ImageType> imgIterator2(img2, img2->GetRequestedRegion());
   while(!imgIterator1.IsAtEnd() && !imgIterator2.IsAtEnd())
   {
      float prediction=imgIterator1.Get();
      float actual=imgIterator2.Get();

      MSE+=std::pow(prediction - actual,2);

      ++imgIterator1;
      ++imgIterator2;
      ++voxelCount;
   }
   std::cout << "MSE=\t" << MSE/voxelCount << std::endl;
}//end of CalculateMSE()


Mat getFeatureVector(ImagePointer WMModStripImg, ImagePointer BRAVOImg, int featuresCount)
{
   std::cout << "generating feature matrix" << std::endl;
	Mat testSample = Mat(1, featuresCount, CV_32FC1); 
	
   MarginateImage(WMModStripImg,5);                        //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.
   MarginateImage(BRAVOImg,5); 

   WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
   BRAVOImg->SetRequestedRegionToLargestPossibleRegion();
   
   itk::ImageRegionIterator<ImageType> inputIterator(WMModStripImg, WMModStripImg->GetRequestedRegion());
   itk::ImageRegionIterator<ImageType> inputIteratorBRAVO(BRAVOImg, BRAVOImg->GetRequestedRegion());  
   
   Mat outMat = Mat(0, 2, CV_32FC1);  
   
   //inputIterator.GoToBegin();
   //inputIteratorB.GoToBegin();
   
   //double threshold=0.3*GetHistogramMax(WMModStripImg,127);
   Mat M = Mat(0, 2, CV_32FC1); 
   
   std::cout << "begin loop" << std::endl;
   
   while(!inputIterator.IsAtEnd()){
		//if(inputIterator.Get() > threshold)                  //the voxels that are not within the threshold (i.e. 0.3 the input image histogram) will be discarded and the corresponding value of 0 will be saved in the segmentation image.
         //{

			//std::cout <<  inputIterator.Get() << "  " << inputIteratorBRAVO.Get()  << std::endl;
			Mat Temp = Mat(1, 2, CV_32FC1);
			Temp.at<float>(0,1)=inputIteratorBRAVO.Get();
			Temp.at<float>(0,0)=inputIterator.Get();
	    	
	    	//std::cout << inputIterator.Get() << " " << inputIteratorBRAVO.Get() << std::endl;

			outMat.push_back(Temp);
			++inputIterator;
			++inputIteratorBRAVO;
	   //}
   }

   

   std::cout << "Done feature production! "<< std::endl;
   return outMat;
}

Mat getLocationVector(ImagePointer WMModStripImg)
{
   	
   MarginateImage(WMModStripImg,5);                        //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.


   WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
   
 
  ImageType::SizeType radius;
  radius[0] = 1;
  radius[1] = 1;
  radius[2] = 1;
  
  itk::NeighborhoodIterator<ImageType> inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
 
   
   
   
   Mat outMat = Mat(0, 5, CV_32FC1);  
   
     
   std::cout << "begin loop" << std::endl;
   
   while(!inputIterator.IsAtEnd()){
		//if(inputIterator.Get() > threshold)                  //the voxels that are not within the threshold (i.e. 0.3 the input image histogram) will be discarded and the corresponding value of 0 will be saved in the segmentation image.
         //{

			//std::cout <<  inputIterator.Get() << "  " << inputIteratorBRAVO.Get()  << std::endl;
			Mat Temp = Mat(1, 5, CV_32FC1);
			Temp.at<float>(0,0)=inputIterator.GetCenterPixel();
			
			
			ImageType::IndexType tempIndex = inputIterator.GetIndex();
			double acc = 0.0;
			double neighbourhood_mean = 0.0;
			int non_zero = 0;
			int c = (inputIterator.Size());
			
			for(int i=0; i<c; i++){		
				double temp  = inputIterator.GetPixel(i);
				
				if(i != (c/2))
				{
					acc += temp;
					if(temp>0){++non_zero;}
				}
			
			}	
			if(non_zero == 0) {
				neighbourhood_mean = 0.0;
			}else{
				neighbourhood_mean = acc/non_zero;
			}
				
			Temp.at<float>(0,1)=neighbourhood_mean;
					
			
			
			Temp.at<float>(0,2)=tempIndex[0];
			Temp.at<float>(0,3)=tempIndex[1];
			Temp.at<float>(0,4)=tempIndex[2];
			
			//std::cout << tempIndex[0] << " " << tempIndex[0] << " " << tempIndex[2] <<  std::endl;;
				    
			outMat.push_back(Temp);
			++inputIterator;
			
	   //}
   }

   

   std::cout << "Done feature production! "<< std::endl;
   return outMat;
}

void ReadSubFolders(std::string folderName, std::string subFolder)
{
   //NOTE: This method is particularly implemented to work with the "folder names", which are being used in the W2MHS toolbox.

   //iterate through all sub-folders of the main directory, which consists of the WM_... images and the pmap images of subjects in separate folders
   //~ std::ifstream list(foldersList);
   //~ if(!list)
   //~ {
      //~ std::cerr << "Cannot read file: " <<  foldersList << std::endl;
      //~ return;
   //~ }
   //~ else
   //~ {
      
         //subfolder="out_..." at this point
         string subFolderID=subFolder;
         string general=folderName+subFolderID;
         subFolderID=subFolderID.erase(0,4);      //keeps the substring after "out_"
         
         std::cout<< general <<std::endl;
         string WMname=general+"/WM_modstrip_"+subFolderID+".nii.gz";
         string PMAPname=general+"/RFREG_pmap_"+subFolderID+".nii.gz";
         string Segoutname=general+"/RFREG_out_"+subFolderID+".nii.gz";
         string featuresName=general+"/trainingFeatures_"+subFolderID+".csv";
         CreateTrainingDataset(WMname,PMAPname,Segoutname,featuresName);

         std::cout << "Training feature set " << subFolder << " created successfully!" << std::endl;
      
   //}
}//end of ReadSubFolders()

void CreateTrainingDataset(std::string WMFilename,std::string  pmapFilename,std::string  segoutFilename,std::string  featuresFilename)
{

   double minNO=0;
   double maxNO=0.699;
   double minYES=0.7;   //0.7 is the gamma value suggested by Kristan. As the RFREG_out images are generated by his modified code, then this threshold will probably works better for creating a training dataset out of W2MHS outputs
   double maxYES=1;
   //
   ImagePointer wmNifti=NiftiReader(WMFilename);
   ImagePointer pmapNifti=NiftiReader(pmapFilename);
   ImagePointer segoutNifti=NiftiReader(segoutFilename);

   MarginateImage(wmNifti,5);      //patch width is 5
   double threshold=0.3*GetHistogramMax(wmNifti,127);      //this is based on what has been done in the W2MHS toolbox
   //1.iterating through the thresholded voxels
   //2.creating patches for each voxel
   //3.extracting features for each patch
   wmNifti->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> inputIterator(wmNifti, wmNifti->GetRequestedRegion());

   pmapNifti->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> pmapIterator(pmapNifti, pmapNifti->GetRequestedRegion());

   segoutNifti->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> segoutIterator(segoutNifti, segoutNifti->GetRequestedRegion());

   while(!inputIterator.IsAtEnd())
   {
      if(inputIterator.Get() > threshold)
      {
         double classLabel=pmapIterator.Get();
//         double segoutVal=segoutIterator.Get();
//         if((segoutVal==1 && classLabel<maxNO) || (segoutVal!=1 && classLabel>minYES))
//         {
//            if(segoutVal==1 && classLabel<maxNO)
//            {
//               classLabel=-1;
//            }
//            if(segoutVal!=1 && classLabel>minYES)
//            {
//               classLabel=1;
//            }



      //         if(classLabel > minNO)
      //         {
      //         if((classLabel > minNO && classLabel <= maxNO) || (classLabel >= minYES && classLabel <= maxYES))
      //         {
      //            if(classLabel > minNO && classLabel <= maxNO)
      //               classLabel=-1;
      //            else    //if(pmapIntensity >= minYES && pmapIntensity <= maxYES)
      //               classLabel=1;

      //               if(classLabel >= 0.5)
      //                  classLabel=1;
      //               else if(classLabel < 0.5)   //if(pmapIntensity >= minYES && pmapIntensity <= maxYES)
      //                  classLabel=-1;
                  ImagePointer patch=GetPatch(wmNifti,inputIterator.GetIndex(), 5);
                  Mat testSample = Mat(1, 2000, CV_32FC1);//NOT BEING USED HERE
                  CreatePatchFeatureVector(patch,testSample,featuresFilename,classLabel);
      //         }
      //         }
//         }
      }
      ++inputIterator;
      ++pmapIterator;
      ++segoutIterator;
   }//end of iteration through the marginated and thresholded input image
}//end of CreateTrainingDataset()

void CreatePatchFeatureVector(ImagePointer patch, Mat patchFeatureMat, std::string  outputFilename,float classLabel)
{
   //NOTE: THIS METHOD OVERLOAD IS USED WHEN IT IS REQUIRED TO SAVE THE FEATURE VECTORS TOGHETHER WITH THEIR CORRESPONDING "CLASSLABEL" INTO A FILE.
   //      THIS IS NOW BEING USED WHEN CREATING A TRAINING DATASET.


   //get all kernels as a list. Kernels should be calculated first and be used as many times as needed.
   std::list<ImagePointer> kernels=GetAllKernels();
   int startIdx=0;      //this is to tell the AppendToPatchFeatureVector(), where to add new features
   //get voxel intensities of each voxel of the path and add them to the feature vector of the patch
   WriteImageFeatureVectorToFile(patch,true,outputFilename);

   for (std::list<ImagePointer>::iterator kernelsIterator=kernels.begin(); kernelsIterator!=kernels.end(); ++kernelsIterator)
   {
      startIdx+=(patch->GetLargestPossibleRegion().GetSize()[0]*patch->GetLargestPossibleRegion().GetSize()[1]*patch->GetLargestPossibleRegion().GetSize()[2]);
      //1. convolve the patch with all the kernels one by one
      //2. get values of the returned convolved image and add them to the feature vector of the patch
      ImagePointer convolvedImg=ConvolveImage(patch,*kernelsIterator,false);
      //NOTE: some scaling has been done on the convolved image in the W2MHS toolbox, which are done here.
      //      When we can get our own ground truth data, we may not need to do these necessarily.
         int kernelIdx=distance(kernels.begin(), kernelsIterator);
         int scalediv=1;
         if(kernelIdx == kernels.size()-1)
            scalediv=3;
         int kernelWidth=kernelsIterator->GetPointer()->GetLargestPossibleRegion().GetSize()[0];
         convolvedImg=DivideImageByConstant(convolvedImg,scalediv*std::pow(kernelWidth,3));

         if(kernelIdx >= 2 && kernelIdx <= 7)
            convolvedImg=AddImageToImage(convolvedImg,patch,false);
         else if(kernelIdx >= 8 && kernelIdx <= 13)
            convolvedImg=AddImageToImage(convolvedImg,patch,true);
      //
      WriteImageFeatureVectorToFile(convolvedImg,false,outputFilename);
   }//end of iterating through kernels list

   //writing the "classLabel at the end of the 'feature vector' row.
   std::ofstream featuresFile;
   featuresFile.open(outputFilename,std::ios::app);
   featuresFile << classLabel << ",";
   featuresFile.close();
}//end of CreatePatchFeatureVector()





