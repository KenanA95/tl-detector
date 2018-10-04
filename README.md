# Traffic Light Detection

Traffic light detection for autonomous driving


For detailed explanations visit medium.com/@kenan.r.alkiek 

## Sample Usage
### Train A Model

1. Customize settings in config.yaml. See config options for more information
2. Run train.py
    
### Run A Model

1. Customize settings in config.yaml. See config options for more information
2. Run run.py


## Config Options


### Train Options
* **descriptor**: hog, lbp, or haar
* **outfile**: Where to save the trained model to

      
***

### Run Options
* **descriptor**: hog, lbp, or haar
* **detector**: spotlight or color
* **heatmap_memory**: # of frames to retain before removing the first frame from memory
* **heatmap_threshold**: # of overlapping detections in the heatmap before accepting a detection 
* **bounding_box_size**: Size of the box to draw on detections