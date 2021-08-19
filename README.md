# Environments

## Analysis Environment

1. All analysis was conducted in a singularity environment using this [definition file](AnalysisEnvironment.def).
2. Exact versions of software used at the time are described in [the package version file](PackageVersions.txt), although exact versions shouldn't matter much.

## Segmentation Environment

1. Segmentation of mice was conducted using code based on the public repository here: [https://github.com/KumarLabJax/MouseTracking](https://github.com/KumarLabJax/MouseTracking)

# Dataset Creation Example Code

## Example Pipeline

1. Run video through segmentation
    - Note: If you use [InfervideoData.py](dataset-creation/InfervideoData.py) from this repository, it will apply the same approach that we used in the paper. This includes scaling the 480x480 network result up to the raspberry pi 1080x1080 video and using a luminance threshold to clean up the mask.
    - Note2: Other segmentation approaches can be used as a drop-in replacement as long as a segmented video is produced.
2. Export image moment data from segmentation using [this code](dataset-creation/ExportScaledSeg.py)
3. Merge segmentation data with eeg/emg annotations into feather format using [this code](dataset-creation/ExportOneAnimal.py)
4. Generate features for classifier using [this code](dataset-creation/Sleep_feature_generation.py)
5. (Optional) Merge multiple animal videos into one for a training dataset.
    - This is achieved simply appending multiple csv files. In linux, you can use a command similar to this: `head -n 1 file1.csv > combined.out && tail -n+2 -q *.csv >> combined.out && mv combines.csv combined.csv`
6. Train a classifier and create predictions on data using [this code](dataset-creation/Sleep_train_classifier.py)
    - Note: Our final classifier used in the paper was trained using the following seed: 1438939568.
7. 

# Analysis Code

## Classifier Performance

1. When training the classifier, performance information will be output.
2. Additionally, example code used in producing the methamphetamine comparisons in the paper is located in [this code](classifier-analysis/performance.py)

## Breathing Rate Plots

1. Plots were created using [this code](breathing-data/AnalyzeStats.py)
2. Merged output file can be found on our Zenodo dataset [here](https://zenodo.org/record/5180680)
3. Merged output file was produced using example code found in [wavelet-analysis](wavelet-analysis/). This code is meant for example resources for creating data used in the paper and will not be supported. The primary wavelet analysis code that is supported is found in the [feature generation code](dataset-creation/Sleep_feature_generation.py)

# Licensing

This code is released under MIT license.

The feature data produced in the associated paper used in training models are released on Zenodo [here](https://zenodo.org/record/5180680) under a Non-Commercial license. Please visit the link for more info.
