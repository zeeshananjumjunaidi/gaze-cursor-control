import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class HeadPoseEstimationModel:

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

    def load_model(self):
        self.plugin = IECore()
        
        model_bin = self.model_name.split(".")[0]+'.bin'
        self.network = self.plugin.read_network(model=self.model_name, weights=model_bin)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)>0 and self.device=='CPU':
            logger.warn("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    logger.error("After adding the extension still unsupported layers found")
                    exit(1)
                logger.info("Extension Added, Issue Resolved!")
            else:
                logger.warn("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = [i for i in self.network.outputs.keys()]
        
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