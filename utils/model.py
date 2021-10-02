import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class ANN_mod:
    def __init__(self,epoch,activation1,activation2,activation3,model_name,LOSS_FUNCTION,OPTIMIZER,METRICS,DL1,DL2,DL3):
        print('ANN_mod')        
        self.epoch = epoch
        self.activation1 = activation1
        self.activation2 = activation2
        self.activation3 = activation3
        self.model_name = model_name                
        self.LOSS_FUNCTION = LOSS_FUNCTION
        self.OPTIMIZER = OPTIMIZER
        self.METRICS = METRICS
        self.DL1  = DL1
        self.DL2 = DL2
        self.DL3 = DL3

    def buid_layer(self):
        print('Layers')
        LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(self.DL1, activation=self.activation1, name="hiddenLayer1"),
          tf.keras.layers.Dense(self.DL2, activation=self.activation2, name="hiddenLayer2"),
          tf.keras.layers.Dense(self.DL3, activation=self.activation3, name="outputLayer")
        ]
        return LAYERS
    def compile_layers(self,model_name1):
        self.model_name1 = model_name1        
        self.model_name1.compile(loss=self.LOSS_FUNCTION, optimizer=self.OPTIMIZER, metrics=self.METRICS)        
        return self.model_name1
    
    def fit_meth(self,X_train,y_train,X_valid,y_valid):
        
        self.X_train = X_train
        self.y_train = y_train        
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        VALIDATION = (self.X_valid,self.y_valid)
        history = self.model_name1.fit(X_train, y_train, epochs=self.epoch, validation_data=VALIDATION)
        print('history.params:',history.params)
        type(history.history)
        print('history_keys:',history.history.keys())
        pd.DataFrame(history.history)
        pd.DataFrame(history.history).plot(figsize=(10,7))
        plt.grid(True)
        plt.show()
    
    def Predict_meth(self,X_new):
        self.X_new  = X_new
        y_prob = self.model_name1.predict(self.X_new)
        y_prob.round(3)
        y_prob.shape
        print(y_prob)
        Y_pred= np.argmax(y_prob, axis=-1)
        return Y_pred,y_prob
    
    
        


        