#! /usr/bin/env python3


class GTHandler:
    def __init__(self, gt_path: str):
        self.path = gt_path

    def split_into_train_val_test(self):
        pass

    

class DataHandler:
    '''
    Sample [N1 / N2 / N3] samples from [gt-train / gt-val / gt-test] 
    for the model on each [train / val / test] run
    '''
    
    def __init__(self):
        pass

    def load_data(self):
        pass

    def preprocess_data(self):
        pass