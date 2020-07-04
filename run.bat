cls
echo "Initializing Gaze Pointer Controller"
@ECHO OFF
REM Sourcing openvino
call "D:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

set FACE_DETECTION_MODEL="models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001.xml"
set GAZE_ESTIMATION_MODEL="models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002.xml"
set HEAD_POSE_ESTIMATION_MODEL="models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001.xml"
set LANDMARKS_REGRESSION_MODEL="models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml"

set FLIP_INPUT_HORIZONTAL="False"
REM set INPUT="cam"
set INPUT="bin\demo-win.mp4"
REM Device to use for for inference, values are "CPU", "GPU", "HETERO:FPGA,CPU"
set DEVICE="CPU"
python src\main.py -d %DEVICE%  -fliph %FLIP_INPUT_HORIZONTAL% -pf -pfl -pge -php -i %INPUT% -f %FACE_DETECTION_MODEL% -g %GAZE_ESTIMATION_MODEL% -hp %HEAD_POSE_ESTIMATION_MODEL% -fl %LANDMARKS_REGRESSION_MODEL%

echo "Program Finished"