echo "Initializing Gaze Pointer Controller"

# Using python version 3.5
python_version=3.5

# Sourcing openvino
source /opt/intel/openvino/bin/setupvars.sh
# Face detection model, FP16
FACE_DETECTION_MODEL="models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
# Using 16FP model for gaze estimation
GAZE_ESTIMATION_MODEL="models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"
# Using 16FP model for head pose estimation
HEAD_POSE_ESTIMATION_MODEL="models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
# Using 16FP model for landmkars regression model
LANDMARKS_REGRESSION_MODEL="models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
# Useful in case of camera is set to flip horizontally.
# To flip camera input horizontally
FLIP_INPUT_HORIZONTAL="True"
# Input stream, Value could be "CAM" or <video file link>
#INPUT="cam"
INPUT="bin/demo.mp4"
# Device to use for for inference, values are "CPU", "GPU", "HETERO:FPGA,CPU"
DEVICE="CPU"

# To learn all parameters of this program run following command
# python3.5 src/main.py --help
python3.5 src/main.py -d $DEVICE  -fliph $FLIP_INPUT_HORIZONTAL -pf -pfl -pge -php -i $INPUT -f $FACE_DETECTION_MODEL -g $GAZE_ESTIMATION_MODEL -hp $HEAD_POSE_ESTIMATION_MODEL -fl $LANDMARKS_REGRESSION_MODEL

