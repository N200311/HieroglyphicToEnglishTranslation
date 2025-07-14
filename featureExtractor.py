# '''
# Created on 5 Jan 2017

# @author: Morris Franken

# Extract deeplearning features from a given image
# '''
# from keras.applications.inception_v3 import InceptionV3
# from sklearn.preprocessing import normalize

# class FeatureExtractor:
#     def __init__(self):
#         print("loading DeepNet (Inception-V3) ...")
#         self.model = InceptionV3(weights='imagenet')
        
#         # Initialise the model to output the second to last layer, which contains the deeplearning featuers  
#         self.model.layers.pop() # Get rid of the classification layer
#         self.model.outputs = [self.model.layers[-1].output]
#         self.model.layers[-1].outbound_nodes = []
     
#     def get_features(self, batch):
#         features =  self.model.predict(batch)
#         features = features.reshape(-1,features.shape[-1])
#         return normalize(features, axis=1, norm='l2') 
'''
Created on 5 Jan 2017

@author: Morris Franken

Extract deeplearning features from a given image
'''
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from sklearn.preprocessing import normalize

class FeatureExtractor:
    def __init__(self):
        print("loading DeepNet (Inception-V3) ...")
        base_model = InceptionV3(weights='imagenet', include_top=True)
        
        # Create a new model that outputs the second to last layer's features
        self.model = Model(inputs=base_model.input,
                          outputs=base_model.layers[-2].output)
     
    def get_features(self, batch):
        features = self.model.predict(batch)
        features = features.reshape(-1, features.shape[-1])
        return normalize(features, axis=1, norm='l2')