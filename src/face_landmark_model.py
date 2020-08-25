import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore
from openvino_model import OpenVINOModel

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class FacialLandmarksDetectionModel(OpenVINOModel):

    def predict(self, image):
        image_to_infer = self.preprocess_input(image.copy())
        infer_dict = {self.input_name:image_to_infer}
        outputs = self.exec_net.infer(infer_dict)

        coords = self.preprocess_output(outputs)

        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        lx,ly,rx,ry = coords
        offset = 20
        left_eye_xmin =  lx-offset
        left_eye_ymin =  ly-offset
        left_eye_xmax =  lx+offset
        left_eye_ymax =  ly+offset   
        right_eye_xmin = rx-offset
        right_eye_ymin = ry-offset
        right_eye_xmax = rx+offset
        right_eye_ymax = ry+offset
        
        left_eye =  image[left_eye_ymin:left_eye_ymax, left_eye_xmin:left_eye_xmax]
        right_eye = image[right_eye_ymin:right_eye_ymax, right_eye_xmin:right_eye_xmax]
        eye_coords = [[left_eye_xmin,left_eye_ymin,left_eye_xmax,left_eye_ymax], [right_eye_xmin,right_eye_ymin,right_eye_xmax,right_eye_ymax]]
        return left_eye, right_eye, eye_coords
        
    def preprocess_input(self, image):
        image_clr_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_clr_convert, (self.input_shape[3], self.input_shape[2]))
        image_t = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return image_t
            

    def preprocess_output(self, outputs):
        outs = outputs[self.output_names][0]
        # Return left eye x, left eye y, right eye x, right eye y
        return (outs[0].tolist()[0][0],outs[1].tolist()[0][0],outs[2].tolist()[0][0],outs[3].tolist()[0][0])
