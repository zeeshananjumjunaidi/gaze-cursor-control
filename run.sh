echo "Initializing Gaze Pointer Controller"

# Using python version 3.5
python_version=3.5
# Sourcing openvino
source /opt/intel/openvino/bin/setupvars.sh

FACE_DETECTION_MODEL="intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
# Using 16FP model for gaze estimation
GAZE_ESTIMATION_MODEL="intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"
# Using 16FP model for head pose estimation
HEAD_POSE_ESTIMATION_MODEL="intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"
# Using 16FP model for landmkars regression model
LANDMARKS_REGRESSION_MODEL="intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml"

# Test Video File
python3.5 src/main.py -d "CPU"  -fliph "False" --previewFlags "fd" "hp" "fld" "ge" -i "bin/demo.mp4" -f $FACE_DETECTION_MODEL -g $GAZE_ESTIMATION_MODEL -hp $HEAD_POSE_ESTIMATION_MODEL -fl $LANDMARKS_REGRESSION_MODEL
# Test Camera
#python3.5 src/main.py -d "CPU" -fliph "False" --previewFlags "fd" "hp" "fld" "ge" -i "CAM" -f $FACE_DETECTION_MODEL -g $GAZE_ESTIMATION_MODEL -hp $HEAD_POSE_ESTIMATION_MODEL -fl $LANDMARKS_REGRESSION_MODEL

#Test Camera Without Preview
#python3.5 src/main.py -d "CPU" -fliph "True" -i "CAM" -f $FACE_DETECTION_MODEL -g $GAZE_ESTIMATION_MODEL -hp $HEAD_POSE_ESTIMATION_MODEL -fl $LANDMARKS_REGRESSION_MODEL