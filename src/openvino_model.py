import cv2
import logging
import numpy as np
import configuration
from openvino.inference_engine import IECore

logger = logging.getLogger()
logger.setLevel(configuration.logType)

class OpenVIINOModel():

    def __init__(self, model_name, device='CPU', extensions=None):
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

        logger.debug("Model Initialized: {}".format(model_name))


    def load_model(self):
        self.plugin = IECore()
        model_bin = self.model_name.split('.')[0]+'.bin'
        self.network = self.plugin.read_network(model=self.model_name, weights=model_bin)

        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        if len(unsupported_layers)>0 and self.device=='CPU':
            logger.warn("Unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                self.plugin.add_extension(self.extensions, self.device)

                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

                if len(unsupported_layers) != 0:
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