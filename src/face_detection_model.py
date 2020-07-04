'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore

#from openvino.inference_engine import IECore

logger = logging.getLogger()
logger.setLevel(configuration.logType)
class FaceDetectionModel:
    
    def __init__(self, model_name, device='CPU', extensions=None,threshold=0.5):
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None
        
        self.device = device
        self.model_name = model_name
        self.extensions = extensions        
        self.prob_threshold = threshold

    def load_model(self):
        self.plugin = IECore()
        model_bin = self.model_name.split('.')[0]+'.bin'
        self.network = self.plugin.read_network(model=self.model_name, weights=model_bin)

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers)>0 and self.device=='CPU':
            logger.warn("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                logger.info("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    logger.error("Even after adding the extension still unsupported layer[s] found")
                    exit(1)
                logger.info("Extension Added, Issue Resolved!")
            else:
                logger.warn("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    def predict(self, image):
        image_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:image_processed})
        coords = self.preprocess_output(outputs, self.prob_threshold)

        if len(coords)==0:
            return 0, 0

        coords = coords[0]

        h,w=image.shape[0],image.shape[1]

        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
      
        cropped_image = image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_image, coords

    def preprocess_input(self, image):
        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return image_processed

    def preprocess_output(self, outputs,probability):
        coords =[]
        data = outputs[self.output_names][0][0]
        for out in data:
            conf = out[2]
            if conf>probability:
                x_min=out[3]
                y_min=out[4]
                x_max=out[5]
                y_max=out[6]
                coords.append([x_min,y_min,x_max,y_max])
        return coords
