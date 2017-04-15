# Blur Kernel Estimation

## Create Datasets
Put all the images you wish to invariantly blur in the `images` folder. Edit according to the channels of the image. Open a MATLAB interpreter and run `create` on MATLAB. This will load the images in the folder and blur each of them with the all different sigma values.

CLI VERSION: `matlab -nodisplay -r create`

This will output two 4-D MATLAB Arrays saved in the `torch` convention (`nImgs` x `nChannels` x `nRows` x `nCols`) of images.

Edit the `create.lua` file according to filenames you'd like to save by. Run `th create.lua`. This output two .t7 storages including shuffling them. 

## Create the Network
Edit `createModel.lua` for the desired depth and size of the network. Run `createModel.lua`. 

## Training & Validating model 
Edit `main.lua` for an appropriate criterion (NLL or MSE). Incase of classification, edit the size of the confusion matrix. 

### Plotting Accuracies
Edit and run `plot.py`. 

## Evaluation
Evaluation is done by predicting the class of every 32x32 patch striding the image by 1px using `evaluator.lua`. It accepts a argument from the CLI and outputs a mat of the predicted sigma map. One can visulize it using the `mesh` command from matlab. Alternatively you may use `savemesh` command in `scripts/` to output files in 3 views.