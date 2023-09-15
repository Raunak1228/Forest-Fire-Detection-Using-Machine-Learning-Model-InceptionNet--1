from keras.models import load_model
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from keras import backend as K

# Load the pre-trained model
model = load_model("C:/Users/HP/Desktop/6_SEM/Forest Fire Detection/model/Newly Trained Model/thermal model.h5")
# Print the list of existing layers in the model
for layer in model.layers:
    print(layer.name)
