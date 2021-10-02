from utils.model import ANN_mod
from utils.all_utils import prepare_data
from utils.all_utils import save_model
from utils.all_utils import save_plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

def main(epoch,Activation1,Activation2,Activation3,LOSS_FUNCTION,OPTIMIZER,METRICS,DL1,DL2,DL3,model_name):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full),(X_test, y_test) = prepare_data(mnist=mnist)
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.

    ANN_mod1 = ANN_mod(epoch=epoch,activation1=Activation1,activation2=Activation2,activation3=Activation3,
                LOSS_FUNCTION=LOSS_FUNCTION,OPTIMIZER=OPTIMIZER,METRICS=METRICS,DL1=DL1,DL2=DL2,DL3=DL3,model_name=model_name)
    LAYERS = ANN_mod1.buid_layer()
    model_name = tf.keras.models.Sequential(LAYERS)
    print('layers:',model_name.layers)
    model_name.summary()
    print('Layer_name:',model_name.layers[1].name)
    weights, biases = model_name.layers[1].get_weights()
    ANN_mod1.compile_layers(model_name1=model_name)


    ANN_mod1.fit_meth(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid)

    model_name.evaluate(X_test, y_test)
    X_new = X_test[:3]
    y_pred,y_prob = ANN_mod1.Predict_meth(X_new=X_new)
    i = 0

    for img_array, pred, actual in zip(X_new, y_pred, y_test[:3]):
        plt.imshow(img_array, cmap="binary")
        plt.title(f"predicted: {pred}, Actual: {actual}")
        plt.axis("off")
        plt.plot()
        i = i + 1
        if i == 1:
            file_name = 'img1.png'
            save_plot(file_name)
        if i == 2:
            file_name = 'img2.png'
            save_plot(file_name)
        if i == 3:
            file_name = 'img3.png'
            save_plot(file_name)

        print("---"*20)
    filename=model_name
    save_model(model_name=model_name,filename=filename)


if __name__ == "__main__":
   main(   
   epoch = 30,
   Activation1= "relu",
   Activation2 = "relu",
   Activation3 = "softmax",
   LOSS_FUNCTION = "sparse_categorical_crossentropy",
   OPTIMIZER = "SGD",
   METRICS = ["accuracy"],
   DL1 = 300,
   DL2 = 100,
   DL3 = 10,
   model_name = "model_clf")



        
