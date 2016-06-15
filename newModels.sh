#!/bin/bash 

#input description #
# -m RF model file
# -w stripped white matter only image .nii
# -g stripped grey matter /CSF only image .nii
# -v ventricle mask .nii
# -s where to save Segmentation
# -p where to save pmap
# -q where to EV quantification
# -c cut off for pmap when doing EV calcs. pixels below cut off not counted. those above used in weighted sum.
# 
# it is assumed that input is in folders for each brain
# .../out_mxxxxxx/WM_modstrip_mxxxxxx.nii, .../out_mxxxxxx/GMCSF_strip_mxxxxxx.nii, .../out_mxxxxxx/Vent_bin_mxxxxxx.nii  



SAMPLE="mxxxxxx" #sample name 
RFMOD="/data/home/uqajon14/w2mhs-itk/envisionRFModel_3107_4242.xml" #presaved RF model file
PMAPCUT="0.5"  #cut-off for pmap for calulating volume 
P_THRESH="0.02"  #cut-off for t-dist model
D_THRESH="1.4"  #cut-off for m-estimator robust normal model
MIN_N="3"  #min. neighbours new models

OUTFOLD="/data/home/uqajon14/Output/" #output base folde, will save like ~/Output/mxxxxxx/. code does NOT create new folders and does not WARN. To be fixed.
BASEFOLD="/data/home/uqajon14/TrainingData/CAI_itk_w2mhs/itk_data/out_"  #input base folder i.e. "~/TrainingData/CAI_itk_w2mhs/itk_data/out_mxxxxxx/"

SAMPLEFOLD=$BASEFOLD$SAMPLE"/"
MYFOLD=$OUTFOLD$SAMPLE"/"

cd ~/w2mhs-itk/build #location of executable

./w2mhs-itk -w $SAMPLEFOLD"WM_modstrip_"$SAMPLE".nii" -g $SAMPLEFOLD"GMCSF_strip_"$SAMPLE".nii" -v $SAMPLEFOLD"Vent_bin_"$SAMPLE".nii" -s $MYFOLD"WMHSeg_"$SAMPLE".nii" -q $MYFOLD"Results_"$SAMPLE".txt" -x $P_THRESH -d $D_THRESH -n $MIN_N


