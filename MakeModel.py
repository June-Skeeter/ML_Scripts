import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# def make_model(GPP_shape,ER_shape,hidden_nodes):
def make_model(Layer_Shapes,hidden_nodes):
    Inputs = []
    Layers = []
    for shape in Layer_Shapes:
        Inputs.append(keras.layers.Input(shape))
        Hidden = keras.layers.Dense(
                                hidden_nodes,
                                activation='relu',
                                kernel_initializer="glorot_uniform",
                                bias_initializer="zeros",
                                )(Inputs[-1])
        Layers.append(
            keras.layers.Dense(
                                1,
                                )(Hidden)
                                )
                              
    Out = keras.layers.Add()(Layers)


    model = keras.models.Model(inputs=Inputs, outputs=Out)


    print(model.summary())
    # keras.utils.plot_model(model, show_shapes=True)
    
    model_json = model.to_json()
    print(model_json)
    # with open(f"temp_files/{name}/model_architecture.json", "w") as json_file:
    #     json_file.write(model_json)
    return(model)


model = make_model([4,2],42)



# keras.utils.plot_model(model, show_shapes=True)