#!/bin/bash 



#input description #
# -m RF model file
# -w stripped white matter only image .nii
# -g stripped grey matter /CSF only image .nii
# -v ventricle mask .nii
# -s where to save Segmentation
# -p where to save pmap
# -q where to EV quantification
# -z do we want to create new model. "new" will trigger new model creation. see -h for other files needed here
# -c cut off for pmap when doing EV calcs. pixels below cut off not counted. those above used in weighted sum.
# 
# it is assumed that input is in folders for each brain
# .../out_m013126/WM_modstrip_m013126.nii, .../out_m013126/GMCSF_strip_m013126.nii, .../out_m013126/Vent_bin_m013126.nii  



SAMPLE="m013126" #sample name 
RFMOD="~/w2mhs-itk/envisionRFModel_3107_4242.xml" #presaved RF model file
PMAPCUT="0.5"  #cut-off for pmap for calulating volume 

OUTFOLD="~/Output/" #output base folde, will save like ~/Output/m013126/. code does NOT create new folders and does not WARN. To be fixed.
BASEFOLD="~/TrainingData/CAI_itk_w2mhs/itk_data/out_"  #input base folder i.e. "~/TrainingData/CAI_itk_w2mhs/itk_data/out_m013126/"

SAMPLEFOLD=$BASEFOLD$SAMPLE"/"
MYFOLD=$OUTFOLD$SAMPLE"/"

cd ~/w2mhs-itk/build #location of executable

./w2mhs-itk -m $RFMOD -w $SAMPLEFOLD"WM_modstrip_"$SAMPLE".nii" -g $SAMPLEFOLD"GMCSF_strip_"$SAMPLE".nii" -v $SAMPLEFOLD"Vent_bin_"$SAMPLE".nii" -s $MYFOLD"WMHSeg_"$SAMPLE".nii" -p $MYFOLD"WMHProb_"$SAMPLE".nii" -q $MYFOLD"Results_"$SAMPLE".txt" -z "nomod" -c $PMAPCUT


