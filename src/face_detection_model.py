'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore
from openvino_model import OpenVINOModel
#from openvino.inference_engine import IECore

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class FaceDetectionModel(OpenVINOModel):
    
    def __init__(self, model_name, device='CPU', extensions=None,threshold=0.5):
        super(FaceDetectionModel,self).__init__(model_name,device,extensions)      
        self.prob_threshold = threshold

    def predict(self, image):
        frame = self.preprocess_input(image.copy())        
        h,w=image.shape[0],image.shape[1]

        outputs = self.exec_net.infer({self.input_name:frame})
        coords = self.preprocess_output(outputs,h,w, self.prob_threshold)

        if len(coords)==0:
            return 0, 0

        first_face = coords[0]
        first_face = np.array(first_face,np.int32)
      
        cropped_image = image[first_face[1]:first_face[3], first_face[0]:first_face[2]]
        return cropped_image, first_face

    def preprocess_input(self, image):
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        frame = np.transpose(np.expand_dims(resized_image,axis=0), (0,3,1,2))
        return frame

    def preprocess_output(self, outputs,height, width, threshold):
        coords =[]
        network_output = outputs[self.output_names][0][0]
        for out in network_output:
            conf = out[2]
            if conf>threshold:
                x_min,y_min=out[3] * width , out[4] * height
                x_max,y_max=out[5] * width , out[6] * height
                coords.append([x_min,y_min,x_max,y_max])
        return coords
