import cv2
import math
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore
from openvino_model import OpenVIINOModel

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class GazeEstimationModel(OpenVIINOModel):

    def predict(self, left_eye_image, right_eye_image, hpa):
        le_image_processed, re_image_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_image_processed, 'right_eye_image':re_image_processed})
        result = self.preprocess_output(outputs,hpa)

        return result

    def preprocess_input(self, left_eye, right_eye):
        if(len(self.input_shape)==4):
            le_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
            re_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
            le_image_processed = np.transpose(np.expand_dims(le_image_resized,axis=0), (0,3,1,2))
            re_image_processed = np.transpose(np.expand_dims(re_image_resized,axis=0), (0,3,1,2))            
            return le_image_processed, re_image_processed
        return False,False
            

    def preprocess_output(self, outputs,hpa):
        gaze_vector = outputs[self.output_names].tolist()[0]
        xdir = -1 if gaze_vector[0]<0 else 1
        ydir = -1 if gaze_vector[1]<0 else 1

        xVal = abs(hpa[0]*0.3)
        yVal = abs(hpa[1]*0.3)

        return (xdir*(abs(gaze_vector[0])+xVal), ydir*(abs(gaze_vector[1])+yVal), gaze_vector[2]),gaze_vector

        