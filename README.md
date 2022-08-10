# Beholder-Interns
This repo serves as repository for all Beholder Intern efforts. \
This repo is dedicated to the regression of distance from monocular imagery. \
To see the original investigation of monocular distance regression, see [this repo](https://gitlab.csde.caci.com/BITS-AI-IRAD/beholder/beholder-ml)

## Install
1. `git clone https://gitlab.csde.caci.com/BITS-AI-IRAD/beholder/beholder-interns.git --recurse-submodules`
2. `cd ngmot` and run one of the download weights scripts to get model weights.
3. Additionally, you will need to download the weights of the distance regressor model and specify the locations in the [cfg_beholder.yml](./cfg_beholder.yml). Update other necessary parameters there too for video input and output options. 
4. Set the model you would like to use in [main_beholder.py](./main_beholder.py).

## Repository Structure
 - DistanceRegressor
    - Contains all Determined.ai experiments and model definitions
 - hololens
    - Contains universal windows platform for the Hololens application
 - notebooks
    - Contains notebooks for evaluating trained model performance, inference timing \
      and regression tree experiments.
 - ngmot 
    - MOT integregration with the distance regressor 
 - assets
    - Folder containing demo video and final IRAD report
 - main_beholder.py
    - Main script for running the end-to-end system

## Research:

This section will discuss various research findings on this topic. 
- Learning Object-Specific Distance From a Monocular Image
    - Propose 2 models: one that takes into account the position of the camera and one does not
    - both models compose of a feature extractor that is fed into a classifier, distance regressor, and a keypoint regressor (if the enhanced model is being used)
    - all models are trained simultaneously with a loss function that adds the loss accross all of the models
    - Dataset: KITTI
    - Results: Abs Rel: 0.251 Squa Rel: 1.844 RMSE: 6.870
        - note that while these results are for the dataset, they are for the vehicle, cyclist and pedestrian classes
    - [Paper](https://arxiv.org/abs/1909.04182)

- Self-Supervised Object Distance Estimation Using a Monocular Camera
    - Used Yolo5v to detect cars and people
    - actual distance is obtained using same method from: Learning Object-Specific Distance From a Monocular Image
        - they used sequential frames, so adding channel spatial and self-attention modules yeilded better results
    - ResNet backbone for distance estimator model
    - Consider camera parameters in training
    - Dataset: KITTI
    - Results: AbsRel: 0.101 SquaRel:0.715 RMSElog: 0.178 
    - [Paper](https://www.mdpi.com/1424-8220/22/8/2936)

## Dataset Info:
- Nuscenes: [Website](https://www.nuscenes.org/nuscenes)
- KITTI: [Github](https://github.com/harshilpatel312/KITTI-distance-estimation)
- InternData: Custom collected data using intern devices and a rangefinder. 
    - Dataset contains images of cars and people and their corresponding distances from the camera. Distances were found using a rangefinder with approximately 1m of error. The data annotations can be found at `/irad_mounts/lambda-quad-5-data/beholder/intern_data/annotations.csv`. The images are located at `/irad_mounts/lambda-quad-5-data/beholder/intern_data/jpg-data`. Note that an annotation distance of -1 corresponds to a car that was annotated, but the distance was not recorded.

## Hardware Plan:
- 1 iPhone 
    - for intern data collection and final streaming of video to MOT
- 1 Tensorbook
    - captures video stream with RTMP server, runs inputs on MOT (dockerized inferencing), and feeds a TCP stream to the hololens.
- 1 Microsoft Hololens
    - for visualizing the end-user output in end-end system

## RTMP Setup (Tensorbook)
Tensorbook:
1. `sudo apt install nginx-core libnginx-mod-rtmp`
2. `sudo nano /etc/nginx/nginx.conf`
3. add this to the bottom of the config file:
```
rtmp {
    server {
        listen 1935;
        chunk_size 4096;
        allow publish [iPhone IP];
        deny publish all;

        application live {
            live on;
            record off;
        }
    }
}
```
> 1935 is the standard RTMP port
4. Save the file
5. `sudo ufw allow 1935/tcp`
6. `service nginx stop`
7. `service nginx start`
8. Make sure the tensorbook is plugged in.

## Iphone Setup
1. Run camera_info.py to get the camera to add to the source code
2. Download the RTMP Live app from the app store. 
3. Open the app and go to settings. 
4. Enter the following:
```
server url: rtmp://[tensorbook IP]:1935/live/
stream name/key: stream
```
5. Adjust settings as necessary.

## Unity / Hololens Setup
1. See the [README](./hololens/README.md)

## Setup Troubleshooting
1. Ensure you are on the same network 
2. Double check the IP of both devices
3. Make sure the camera is at the height of the hololens and start both in the same direction

## Lessons Learned:
- Average cellular GPS accuracy has an uncertainty of ~5m radius. See [here](https://www.gps.gov/systems/gps/performance/accuracy/)
- Streaming sites often throttle 


## Geo-Rectify TODO
get lla of camera
convert lla to ecef of camera
estimate distance to target
need either azimuth in 2d and bearing to find direction that target lies from camera
add camera coordinates to target coordinates
send target coordinates to unity

unity side:
subtract hololens coords (convert lla to ecef) from target coords
place object in unity environment at those coordinates
also somehow plot these with consideration for the bearing/direction of the hololens

maybe strap one of these to camera and one to hololens to stream that info? Gives GPS too!
https://www.sparkfun.com/products/15712