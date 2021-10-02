import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def prepare_data(mnist):
    (X_train_full, y_train_full),(X_test, y_test) = mnist.load_data()
    X_train_full.shape
    X_test.shape
    X_train_full[0].shape
    img = X_train_full[0]
    plt.imshow(img, cmap="binary")
    plt.axis("off")
    plt.show()
    y_train_full.shape
    y_train_full[0]
    plt.figure(figsize=(20,20))
    sns.heatmap(img/255, annot=True, cmap="binary")
    return (X_train_full, y_train_full),(X_test, y_test)



def save_plot(file_name):
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    plotPath = os.path.join(plot_dir, file_name)  # model/filename
    plt.savefig(plotPath)