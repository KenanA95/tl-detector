# Traffic Light Detection

Traffic light detection for autonomous driving


For detailed explanations visit [medium.com/@kenan.r.alkiek](http://medium.com/@kenan.r.alkiek) and docs/

## Dependencies
* cv2
* numpy
* sklearn
* skimage
* matplotlib
* yaml

## Sample Usage
### Train A Model

1. Customize settings in config.yaml (see config options for more information)
2. Run train.py
    
### Run A Model

1. Customize settings in config.yaml (see config options for more information)
2. Run run.py


## Config Options


### Train Options
* **descriptor - _str_**: hog, lbp, or haar
* **classifier - _str_**: svm (more options to be added)
* **outfile - _str_**: File location to save the trained model to
* **positive_image_directory - _str_**: Where the positive images are stored
* **negative_image_directory - _str_**: Where the negative images are stored
* **window_size - _tuple (int, int)_**: Size of the training windows

      
### Run Options
* **descriptor - _str_**: hog, lbp, or haar
* **classifier_location - _str_**: Location of the saved classifier (the output of running train.py)
* **detector - _str_**: spotlight, color
* **heatmap_memory - _int_**: Number of frames to retain before removing the first frame from memory
* **heatmap_threshold - _int_**: Number of overlapping detections in the heatmap before accepting a detection 
* **window_size - _tuple (int, int)_**: Size of the boxes to classify and draw on
* **image_directory - _str_**: Image directory to run the classifier on

### Descriptor Options

#### HOG
* **block_size - _tuple (int, int)_**:  Number of cells in each block
* **cell_size - _tuple (int, int)_**: Size (in pixels) of a cell
* **orientations - _int_**: Number of orientation bins

#### LBP
* **points - _int_**: Number of circularly symmetric neighbour set points 
* **radius - _float_**: Radius of circle (spatial resolution of the operator)

#### Haar
* **selected_feature_file - _str_**: File containing the coordinates and types of the Haar features you want to use. More information in the docs


### Detector Options

#### Spotlight
* **max_size - _int_**: How large a spotlight can grow before being rejected
* **kernel - _int_**: Size of the kernel to apply during top-hat morphology
* **threshold - _int_**: Threshold value applied after top-hat morphology
