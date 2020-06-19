echo "Initializing Gaze Pointer Controller"

# Using python version 3.5
python_version=3.5
# Sourcing openvino
source /opt/intel/openvino/bin/setupvars.sh

FACE_DETECTION_MODEL="model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
# Using 16FP model for gaze estimation
GAZE_ESTIMATION_MODEL="model/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"
# Using 16FP model for head pose estimation
HEAD_POSE_ESTIMATION_MODEL="model/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"
# Using 16FP model for landmkars regression model
LANDMARKS_REGRESSION_MODEL="model/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml"
# Useful incase of camera is set to flip horizontally.
FLIP_INPUT_HORIZONTAL="False"
# Input stream its value could be "CAM" or <video file link>
INPUT="bin/demo.mp4" #"CAM"
# Device to use for for inference, values are "CPU", "GPU", "HETERO:FPGA,CPU"
DEVICE="CPU"

python3.5 src/main.py -d $DEVICE  -fliph $FLIP_INPUT_HORIZONTAL --previewFlags "fd" "hp" "fld" "ge" -i $INPUT -f $FACE_DETECTION_MODEL -g $GAZE_ESTIMATION_MODEL -hp $HEAD_POSE_ESTIMATION_MODEL -fl $LANDMARKS_REGRESSION_MODEL

