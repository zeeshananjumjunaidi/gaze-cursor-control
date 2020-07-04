import cv2
import math
import logging
import numpy as np
from openvino.inference_engine import IECore

logger = logging.getLogger()

class GazeEstimationModel:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        self.plugin = IECore()        
        model_bin = self.model_name.split(".")[0]+'.bin'
        self.network = self.plugin.read_network(model=self.model_name, weights=model_bin)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0 and self.device=='CPU':
            logger.warn("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                logger.info("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    logger.error("After adding the extension still unsupported layers found")
                    exit(1)
                logger.info("After adding the extension the issue is resolved")
            else:
                logger.warn("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

        
    def predict(self, left_eye_image, right_eye_image, hpa):
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        result = self.preprocess_output(outputs,hpa)

        return result

    def preprocess_input(self, left_eye, right_eye):
        if(len(self.input_shape)==4):
            le_image_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
            re_image_resized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
            le_img_processed = np.transpose(np.expand_dims(le_image_resized,axis=0), (0,3,1,2))
            re_img_processed = np.transpose(np.expand_dims(re_image_resized,axis=0), (0,3,1,2))            
            return le_img_processed, re_img_processed
        return False,False
            

    def preprocess_output(self, outputs,hpa):      
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        xdir = -1 if gaze_vector[0]<0 else 1
        ydir = -1 if gaze_vector[1]<0 else 1

        xVal = abs(hpa[0]*0.5)
        yVal = abs(hpa[1]*0.5)

        return (xdir*(abs(gaze_vector[0])+xVal), ydir*(abs(gaze_vector[1])+yVal), gaze_vector[2]),gaze_vector

        