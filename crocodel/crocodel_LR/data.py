import numpy as np
import yaml
from . import ll

class Data:
    """Data class to contain and analyse typical high-resolution cross-correlation spectroscopy dataset.
    """
    def __init__(self, *args, **kwargs):
        
        ## Load the information from the croc_config.yaml file taken as input when initiating this class.
        # with open(kwargs.pop('config_dd')) as f:
        #     self.config = yaml.load(f,Loader=yaml.FullLoader)
        self.config = kwargs.pop('config_dd')
        self.inst_list = self.config['inst_list'].keys()
        self.inst = {}
        for inst_name in self.inst_list:
            self.inst[inst_name] = ll.Instrument_LR(inst_name, self.config['inst_list'][inst_name])
            
        #### Load the data for each intrument, and use ll.py to compute the log likelihood 