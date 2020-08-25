import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore
from openvino_model import OpenVINOModel

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class HeadPoseEstimationModel(OpenVINOModel):

    def predict(self, image):
        image_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:image_processed})
        result = self.preprocess_output(outputs)
        return result
        
    def preprocess_input(self, image):
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return image_processed
            

    def preprocess_output(self, outputs):
        outs = [
			outputs['angle_y_fc'].tolist()[0][0],
   outputs['angle_p_fc'].tolist()[0][0],
   outputs['angle_r_fc'].tolist()[0][0]
		]
        return outs
