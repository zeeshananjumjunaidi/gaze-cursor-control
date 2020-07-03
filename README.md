# Computer Pointer Controller

Gaze Cursor Control program is designed to controll mouse pointer using realtime gaze input. This will also include the head orientation. User can provide either video or live camera streaming as input.

## Project Set Up and Installation

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

Edit run.sh and update the path of models.

Open terminal and navigate to project root directory.
Then run ```./run.sh```
You can edit run.sh to change parameteres, like if you want to provide video file input or use model from different location etc...

Paramters can be changed in run.sh
All variables are self explanatory.

- **FACE_DETECTION_MODEL** \<Path of xml file of face detection model>
- **LANDMARKS_REGRESSION_MODEL** \<Path of xml file of facial landmarks detection model>
- **HEAD_POSE_ESTIMATION_MODEL** \<Path of xml file of head pose estimation model>
- **GAZE_ESTIMATION_MODEL** \<Path of xml file of gaze estimation model>
- **INPUT** \<Path of video file or camera, input values are either link of video file or "CAM" for camera input> 
- **DEVICE** \<Use processing unit for inference, values are CPU, GPU, or HETERO:FPGA,CPU>

## Documentation


### Models Documentation:
- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

### Paramters

```
usage: main.py [-h] -f FACEDETECTIONMODEL -fl FACIALLANDMARKMODEL -hp
               HEADPOSEMODEL -g GAZEESTIMATIONMODEL -i INPUT
               [-flags PREVIEWFLAGS [PREVIEWFLAGS ...]] [-l CPU_EXTENSION]
               [-prob PROB_THRESHOLD] [-d DEVICE] -fliph FLIP_HORIZONTAL

optional arguments:
  -h, --help            show this help message and exit
  -f FACEDETECTIONMODEL, --facedetectionmodel FACEDETECTIONMODEL
                        Specify Path to .xml file of Face Detection model.
  -fl FACIALLANDMARKMODEL, --faciallandmarkmodel FACIALLANDMARKMODEL
                        Specify Path to .xml file of Facial Landmark Detection
                        model.
  -hp HEADPOSEMODEL, --headposemodel HEADPOSEMODEL
                        Specify Path to .xml file of Head Pose Estimation
                        model.
  -g GAZEESTIMATIONMODEL, --gazeestimationmodel GAZEESTIMATIONMODEL
                        Specify Path to .xml file of Gaze Estimation model.
  -i INPUT, --input INPUT
                        Specify Path to video file or enter cam for webcam
  -flags PREVIEWFLAGS [PREVIEWFLAGS ...], --previewFlags PREVIEWFLAGS [PREVIEWFLAGS ...]
                        Specify the flags from fd, fld, hp, ge like --flags fd
                        hp fld (Seperate each flag by space)for see the
                        visualization of different model outputs of each
                        frame,fd for Face Detection, fld for Facial Landmark
                        Detectionhp for Head Pose Estimation, ge for Gaze
                        Estimation.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to detect the face
                        accurately from the video frame.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -fliph FLIP_HORIZONTAL, --flip_horizontal FLIP_HORIZONTAL
                        Flip input horizontally
```

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

Average result

|            Name           | Model Load Time FP32 | Inference time (FP32) |
|:-------------------------:|:------------:|:--------------------------:|
| Face detection            | 0.172        | 0.008                      |
| Face Landmark detection   | 0.057        | 0.0007                      |
| Head Pose Estimation      | 0.062        | 0.002                      |
| Gaze Estimation           | 0.080        | 0.001                      |
| **Total Average**             | 0.092          |0.003                      |

|            Name           | Model Load Time FP16 | Inference time (FP16) |
|:-------------------------:|:------------:|:------------------------:|
| Face detection            | 0.348           | 0.013                      |
| Facial Landmark detection | 0.105        | 0.0001                    |
| Head Pose detection       | 0.173        | 0.002                    |
| Gaze Estimation           | 0.170       | 0.003                    |
| **Total Average Inference**   | 0.199       | 0.004                    |



|   Model Type  |   FPS     |
|:----------:|:------------:|
|   FP32        |  ~13FPS    |
|   FP16        |  ~12FPS         |


## Results

I tested this program on Intel i7 (7700K) VM. From the results, it seems like FP16 too long time for loading, but for i7 (7700K) loading time is low and there is not much difference in model inference as FPS are almost similar in both type of the models.
Higher floating point precision usually have higher accuracy.
Normally precision model usually takes more time for inference than lower precision models.
From the result, we can see that loading time of FP16 is slower than the FP32 by 10ms.
We prefer using lower precision model when we have constraints such as low power consumption and less processing power.

<hr/>
### Testing face detection model on different hardware.

I tested Face detection model on multiple devices.
- CPU - i5-6500te:iei-mustang-f100-a10  - FP32
- IGPU - i5-6500te:intel-hd-530 - FP32
- VPU - intel-ncs2 - FP32
- FPGA - iei-mustang-f100-a10 - FP16

#### Model Loading Time
![Model Loading Time](./images/model-loading-time.png)

#### Inference Time 
![Model Inference Time](./images/model-inference-time.png)

#### FPS
![Frame Per Seconds](./images/fps.png)

### Edge Cases
1. If program unable to find face in video input, it will print 'Unable to detect the face' and continue to read another frame.
2. Model will use only one face detected in the streaming.
