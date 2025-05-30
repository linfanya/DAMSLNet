# DAMSLNet:Dual-AttentionMulti-ScaleLightweightNetworkforPlantDiseaseClassification

## Datasets:

* The xinong apple dataset at https://aistudio.baidu.com/aistudio/datasetdetail/11591.
It is created by Northwest University of Agriculture and Forestry Science and Technology. This dataset contains 8 types of apple leaf diseases. Most of the pictures in the dataset were taken by mobile phones at the Baishui, Luochuan and Qingcheng Apple Test Stations in China in 2021. In addition, a small amount of open-source disease data is also integrated. The dataset is mainly obtained under the condition of good light on sunny days, and some of it is collected on rainy days. Different collection conditions further enhance the diversity of the dataset. The background of the image includes the laboratory background and the complex agricultural environment.

* The FGVC8 dataset at https://pan.baidu.com/s/1W8aI1tObO8t02Z7zkKWvGA?pwd=2wxr.
It's an Apple disease dataset built by Cornell Initiative for Digital Agriculture (CIDA) in 2021. Apple is one of the most important temperate fruit crops in the world. Leaf disease poses a major threat to the overall productivity and quality of apple orchards. The dataset significantly increased the number of apple leaf disease images on the basis of FGVC7, and added additional disease categories. The dataset has about 23,000 high-quality RGB images, which are divided into 12 categories. It reflects the real scene by showing the non-uniform background of leaf images taken at different maturity stages and at different times of the day with different focal length camera settings.

* The Plantvillage dataset at https://pan.baidu.com/s/1DinVIDPlr_WZ3GeK7B2a8g?pwd=9zn2.
The dataset collected 54,303 health and disease images under controlled conditions, divided into 38 disease categories. These pictures cover 14 crops, including apple, blueberry, cherry, grape, orange, peach, pepper, potato, raspberry, soy, squash, strawberry and tomato. It contains images of 17 basic diseases, 4 bacterial diseases, 2 diseases caused by mold (oomycete), 2 viral diseases and 1 disease caused by mites. Images of 12 healthy crop leaves will not be significantly affected by the disease.

* The Rice dataset at https://pan.baidu.com/s/1IDGUFws2HDy72jxevUzLDg?pwd=mrq3.
This dataset contains 5,932 pictures, 4 types of rice leaf diseases, including Bacterialblight, blast, Brownspot and Tungro. Most of the dataset was taken by Nikon cameras in different rice fields in western Orissa, India, and a small number were taken from the agricultural disease and pest picture database. The samples in the data set are rich, covering single and complex background samples required for experiments.

## Introduction

* If you want to use our code, you must have the following preparation under the PyTorch framework: see requirement.txt for details.

## Code Guidance:
* Download the dataset in the above link, put the training images and labels into your specify path, then run split_data.py file to divide the dataset into the training and testing set, then you can run the data_appand.py file to preprocess the datasets. Here are several methods for enhancing data for reference.
* After that, you can run the train.py file for training and run test.py to test the network. The accuracy and loss value of training and testing set will be displayed on the terminal. 
* Running the plot_confuse.py file will generate confusion matrix and ROC curve. By using the confusion matrix, accuracy, precision, recall, and F1-Score can be obtainedand be displayed on the terminal. Running the heatmap.py and featuremap.py will generate the corresponding heatmap and featuremaps.

## Reference

* Linfan Deng, Juan Qin, et al. DAMSLNet:Dual-AttentionMulti-ScaleLightweightNetworkforPlantDiseaseClassification (2025).
