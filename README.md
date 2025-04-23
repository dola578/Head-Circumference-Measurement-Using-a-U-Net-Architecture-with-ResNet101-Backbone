# HC-18 Grand Challenge: Automatic Measurement of Fetal Head Circumference Using a U Net Architecture with ResNet101 Backbone
# Introduction
This repository includes the source codes and raw datasets of an Automatic Measurement of Fetal Head Circumference approach using a U Net Architecture with ResNet101 Backbone. This is an attempted solution to the [HC-18 Grand Challenge.](https://hc18.grand-challenge.org/) 

Fetal Biometric Assessments are essential for monitoring fetal development and maternal health during and after pregnancy. One of the most significant biological characteristics of a fetus is head circumference (HC). The common clinical practice is to manually measure the HC from ultrasound images. The cons of manual appraoch are that they are error-prone, issues with low contrast, fuzzy or missing borders, speckle noise in US images, variability in image quality leading to inconsistent results, low-resource settings or limited domain expertise etc. An automated system can reduce scan time and minimize repetitive manual tasks, enhance reliability and reproducibility of fetal measurements, and replace sonographers with less expertise.

I proposed developing a robust segmentation model using U-Net integrated with ResNet101 for feature extraction to effectively segment the fetal skull in ultrasound images, combining Binary Cross-Entropy (BCE) loss and Dice coefficient loss to address challenges of class imbalance in medical image segmentation. For refining segmentation results, image processing techniques such as morphological operations and post-segmentation were implemented to reduce noise and correct imperfections in the segmented boundaries. The overall workflow is illustrated below:

![Workflow](https://github.com/user-attachments/assets/9e2dd6fe-8c45-4f93-965e-6148d2de4579)

# Architecture

![Unet with Resnet101](https://github.com/user-attachments/assets/cc11961d-9d76-4005-8bd9-dc0e92a41510)

# **Dataset**
The dataset can be accessed from [here](https://webmailuwinnipeg-my.sharepoint.com/:f:/r/personal/dola-s_webmail_uwinnipeg_ca/Documents/Fetal%20Head%20Circumference%20dataset?csf=1&web=1&e=mBKQLT). The data is provided by the HC-18 challenge. The datasets are already divided into training and test sets. There are 999 images with 999 annotations in the training set and 335 images only in the test set. A csv file containing the pixel sizes and the actual HC measurements for each training sample is provided, while another csv file with just the pixel sizes of the test samples is also given. Since it is a challenge, the test data labels are not provided by the challenge team.

> [!NOTE]
>The challenge team is currently not accepting submissions as they are facing problems from their end and have not fixed the issue yet. Therefore, from my 999 training data, I kept 749 images as training data and 250 images as test data.

# Library Requirements
> [!NOTE]
>A specific version of the typing_extensions is required to avoid errors in the code. This can be run for the typing_extensions:
>```
>!pip install typing_extensions==4.12.2 --upgrade
>```
>or
>```
>!pip install typing_extensions==4.7.1 --upgrade
>```
>If it still gives some errors, then this line can be run before running the above ones:
>```!pip install typing_extensions>=4.3 --upgrade```

For the other libraries given below, the latest version will suffice.
```
torchvision
PyWavelets
opencv-python
```

# Preprocessing
## Denoising using Wavelet Transform
Both the training data and the testing data were denoised using wavelet transform. The ```Denoising_all_data.ipynb``` can be run twice - once with train data as input, and again with test data as input. 2 new directories named **denoised_train_set** (without annotations) and **denoised_test_set** will be created.

## Masking Annotations
After that, the annotations of the training data were filled/masked using Flood Fill algorithm. The ```masking_annotations.ipynb``` can be run and the masked annotations will be stored in a directory named ```masked_annotations.```
![annotations (1)](https://github.com/user-attachments/assets/3c5d9e1d-386e-4df4-bf0b-291ac651c7ec)

## Flipping, Resizing, Rotation, Intensity-based Transformation
Moving on, the ```Preprocessing - flipping, resizing, rotation, intensity based transformation``` section from ```HC.ipynb``` can be run to perform flipping, resizing (256x256), rotation and some intensity based transformation on the training data.

# Training
1. The ```Training Loop``` section in ```HC.ipynb``` can be run for training the model. Run the ```Unet with resnet101 as backbone``` and the ```Dice loss + Binary cross-entropy loss``` section in the same python file first. These are the parameters used in the training loop:
  + ```optimizer```: Utilizes Adam optimizer
  + ```scheduler```: Configured to monitor the 'min' of a monitored quantity, reduce the learning rate by a factor of 0.1 after ```patience``` of 10 epochs of no improvement.
  + ```num_epochs```: Epoch size set to 50
  + ```device```: CUDA. If not available, then cpu.
2. Parameters for the model (ResNet-101 backbone with U-Net structure):
  + ```base_model```: Pre-trained ResNet-101 as the encoder.
    + ```pretrained=true```: pre-trained on ImageNet.
  + ```Encoder Blocks from ResNe-101```
    + ```self.layer0```: The initial set of layers in ResNet-101 that includes the first convolution, batch normalization, and ReLU activation.
    + ```self.maxpool```: Max pooling layer to reduce spatial dimensions.
    + ```self.layer1```: This is the first of the four main blocks of ResNet-101, outputting 256 feature channels.
    + ```self.layer2```: The second block, outputting 512 feature channels.
    + ```self.layer3```: The third block, outputting 1024 feature channels.
    + ```self.layer4```: The final block of the encoder, outputting 2048 feature channels.
  + ```Decoder```: Each block consists of an upsampling step followed by a convolution block to refine features and restore spatial dimensions.
    + ```self.up4```: Upsamples and combines features from layer3 (1024 channels) and layer4 (2048 channels) to give 1024 output channels.
    + ```self.up3```: Upsamples and combines features from layer2 (512 channels) and the output of up4 to produce 512 output channels.
    + ```self.up2```: Further upsampling and combination of layer1 (256 channels) and the output of up3 to give 256 channels.
    + ```self.up1```: The final upsampling step that combines the very early features from layer0 (64 channels) and the output of up2 to restore to 64 channels.
  + ```Final Layers```
    + ```self.final_up```:  An additional upsampling layer to scale the output back to the input image size.
    + ```self.final_conv```: A convolutional layer with a kernel size of 1, used to map the deep features to the desired number of output classes (n_classes), typically set for binary segmentation.
    + ```Activation Function```: A sigmoid activation function applied to the final convolutional output to map the logits to probabilities.
3. For Loss Function, a combination of Dice Loss and Binary Cross-Entropy was used.

5 fold validation was performed during training. Random images with the predicted segmented overlay are generated during training, and kept in overlay_images directory.

# Model
The model was saved as ```best_unet_resnet101_model.pth```. The saved model can be found [here](https://webmailuwinnipeg-my.sharepoint.com/:f:/r/personal/dola-s_webmail_uwinnipeg_ca/Documents/Fetal%20Head%20Circumference%20dataset?csf=1&web=1&e=mBKQLT).

# Test
The model was then run on the test data for getting segmented test data images. The ```Running Model on test data``` section of ```HC.ipynb``` can be run, which will store the segmented images in **output_segmentations** directory. 

# Evaluating segmentation
For evaluating the segmentation, DSC and HD were calculated with filled annotations as the ground truth. The ```DSC (Dice Similarity Coefficient) with filled annotations as ground truth``` section of ```HC.ipynb``` can be run to get the DSC and HD values.
![segmentation](https://github.com/user-attachments/assets/299e54be-6e16-4980-9194-31283f101d88)

# Post Segmentations
Morphological opening closing and canny edge detector was used for post segmentation steps. The ```Morphological Opening and Closing + Canny edge Detector``` section from ```HC.ipynb``` can be run which will store the segmented edges in **output_edges.**

# Ellipse fitting
Ellipse fitting was used on the edges to get the HC parameters, from which HC can be calculated. fitEllipse from OpenCV was used here. The ```Ellipse fitting``` section of the ```HC.ipynb``` can be run, which will generate a csv file with the HC parameters (c_x, c_y, semi_a, semi_b, angle).

# Calculating Head Circumference (HC)
I used Ramanujan's approximation formula for calculating HC. The code block sections under ```Calculating HC``` can be run to get the predicted calculations.

**HC = π × [ 3(a + b) − √((3a + b)(a + 3b)) ]**

Where:
- *a* = semi-major axis (in mm)
- *b* = semi-minor axis (in mm)

# Evaluating HC (Head Circumference)
For evaluation, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²), Mean Difference (DF) and Mean Absolute Difference (ADF) were used. The code block sections under ```Evaluation``` can be run to get the evaluation metrics values.


