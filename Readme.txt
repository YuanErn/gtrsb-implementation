TRAFFIC SIGN CLASSIFICATION – COMP30027 PROJECT 2 (2025)
=========================================================

Overview:
---------
This project aims to classify German traffic signs into one of 43 classes using extracted features. 


Directory Structure:
--------------------
The project is organized as follows:



- train\
    Folder with training image
	- train_metadata.csv
    		Metadata for training images (ids, image filenames and class labels)
	- Features\
    		Contains:
      		- color_histogram.csv
      		- hog_pca.csv
      		- additional_features.csv


- test\
    Folder with test images 
	- test_metadata.csv
    		Metadata for test images (ids, image filenames only, no labels)
	- Features\
    		Contains:
      		- color_histogram.csv
      		- hog_pca.csv
      		- additional_features.csv

- README.txt
    This file

Data:
-----
- 5488 training images with class labels
- 2353 test images without labels
- 43 total traffic sign classes
- Provided features:
    * HOG (Histogram of Oriented Gradients) - PCA reduced
    * Color histograms
    * Additional features (edge density, texture variance, mean RGB)


Submission Format (Kaggle) (CSV):
------------------------
Final submission should follow this structure:

Id,ClassId
67.jpg,4
94,2
...
521.jpg,12