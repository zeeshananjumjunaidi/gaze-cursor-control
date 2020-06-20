# Computer Pointer Controller

Gaze Cursor Control program is designed to controll mouse pointer using realtime gaze input. This will also include the head orientation. User can provide either video or live camera streaming as input.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
### Step: 1
Clone the repository: https://github.com/zeeshananjumjunaidi/gaze-cursor-control
### Step: 2
Initialize the openVINO environment:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
### Step 3
Download following models by using OpenVINO model downloader

- **1. Face Detection Model**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
- **2. Facial Landmarks Detection Model**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```
- **3. Head Pose Estimation Model**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```
- **4. Gaze Estimation Model**
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

## Demo

Open terminal and navigate to project root directory.
and run ```./run.sh```
You can edit run.sh to change parameteres, like if you want to provide video file input or use model from different location etc...

Paramters can be changed in run.sh
All variables are self explanatory.

- FACE_DETECTION_MODEL \<Path of xml file of face detection model>
- LANDMARKS_REGRESSION_MODEL \<Path of xml file of facial landmarks detection model>
- HEAD_POSE_ESTIMATION_MODEL \<Path of xml file of head pose estimation model>
- GAZE_ESTIMATION_MODEL \<Path of xml file of gaze estimation model>
- INPUT \<Path of video file or camera, input values are either link of video file or "CAM" for camera input> 
- DEVICE \<Use processing unit for inference, values are CPU, GPU, or HETERO:FPGA,CPU>

## Documentation


### Models Documentation:
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

### Project Structure

- .run.sh Main file to run the Gaze Cursor Control pipeline. 
- src folder contains all models defination, including input feeder, mouse controller, and main pipeline to run.
- src/main.py Main pipeline file to run this project. This require input of models, video stream, and device to be used for inference.
- src/mouse_controller.py this file contain MouseController class which take x,y coords., speed, and precision and set those values to actual mouse pointer.
- src/face_detection_model.py Used for face detection
- src/face_landmark_model.py Used for detecting eyes in a given face.
- src/head_pose_model.py Used for detecting Head pose/ orientation.
- src/gaze_estimation_model.py Used for gaze prediction given left, right eyes and head pose angles.
- src/input_feeder.py Contains InputFeeder class to initialize camera and return frame sequentially.
- models folder contains models provided by the intel for face detection, landmark detection, gaze estimation, and head pose prediction.
- media folder contains demo input files for testing this program.

## Benchmarks

*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
1. If program unable to find face in video input, it will print 'Unable to detect the face' and continue to read another frame.
2. Model will use only one face detected in the streaming.
