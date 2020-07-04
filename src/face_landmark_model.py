import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class FacialLandmarksDetectionModel:

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
        
        
        if len(unsupported_layers)>0 and self.device=='CPU':
            logger.warn("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                logger.info("Adding cpu_extension")
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
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        
    def predict(self, image):
        image_processed = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:image_processed})
        coords = self.preprocess_output(outputs)
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        le_xmin=coords[0]-20
        le_ymin=coords[1]-20
        le_xmax=coords[0]+20
        le_ymax=coords[1]+20
        
        re_xmin=coords[2]-20
        re_ymin=coords[3]-20
        re_xmax=coords[2]+20
        re_ymax=coords[3]+20

        left_eye =  image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin,le_ymin,le_xmax,le_ymax], [re_xmin,re_ymin,re_xmax,re_ymax]]
        return left_eye, right_eye, eye_coords
        
    def preprocess_input(self, image):
        image_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_convert, (self.input_shape[3], self.input_shape[2]))
        image_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return image_processed
            

    def preprocess_output(self, outputs):
        outs = outputs[self.output_names][0]
        leye_x = outs[0].tolist()[0][0]
        leye_y = outs[1].tolist()[0][0]
        reye_x = outs[2].tolist()[0][0]
        reye_y = outs[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)