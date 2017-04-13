# Blur Kernel Estimation

## Create Datasets
Put all the images you wish to invariantly blur in the `images` folder. 
Open a MATLAB interpreter and run `create` on MATLAB. This will load the images in the folder and blur each of them with the all different sigma values.

CLI VERSION: `m2 -nodisplay -r 'try create; catch; end; quit'`

This will output two 4-D MATLAB Arrays saved in the `torch` convention (`nImgs` x `nChannels` x `nRows` x `nCols`) of images.

Edit the `create.lua` file according to filenames you'd like to save by. Run `th create.lua`. This output two .t7 storages including shuffling them. 

## Create the Network
Edit `createModel.lua` for the desired depth and size of the network. Run `createModel.lua`. 

## Training & Testing 


### Plotting Accuracies
Edit and run `plot.py`. 


