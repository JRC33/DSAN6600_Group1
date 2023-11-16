
### Package Imports ###
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import cv2
import skimage.measure
import PIL
from PIL import Image
import torch



def main():
    #page header
    st.markdown('H.A.L. 9000')

    #bring in models
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
    klaasmodel = tf.keras.models.load_model('../model_file/sign_language_model.keras')

    #remove runs folder just in case 
    

    imageCaptured = st.camera_input("Do some sign language!",key="FirstCamera", help ="Do some sign language!")

    if imageCaptured is not None:
        
        
        img = Image.open(imageCaptured)

        results = model_yolo(img)
        results.save()

        img_array = np.array(img)

        

        st.markdown(img_array.shape)
        
        img_array = img_array.mean(axis = 2 , keepdims = True)

        st.markdown(img_array.shape)
        step_size_i = img_array.shape[0] // 28
        step_size_j = img_array.shape[1] // 28

        img_array = skimage.measure.block_reduce(img_array, (step_size_i, step_size_j, 1), np.mean)


         #if the shape is not divisible by 28, we have dimensions of size 29
        img_array = img_array[:28, :28, :]

        # to send to model, we need the batch dimension as well
        img_array = np.expand_dims(img_array, 0)

        st.markdown(img_array.shape)


        






   










if __name__ == '__main__':
    main()









