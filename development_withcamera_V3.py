
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
import torchvision
import IPython
import shutil

st.set_page_config(layout="wide")
st.session_state['textlist'] = 0


def increment(imageCaptured):
    if 'count' not in st.session_state:
        st.session_state.count = []

    
    #st.session_state.count.append('test')
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
    klaasmodel = tf.keras.models.load_model('../model_file/sign_language_model.keras')

    img = Image.open(imageCaptured)

    os.system('rm -rf runs')

    results = model_yolo(img)
    results.save()


    #st.image('runs/detect/exp/image0.jpg' , width = 300)

    result = results.pandas().xyxy[0]
    xmin, xmax, ymin, ymax = int(result['xmin']), int(result['xmax']), int(result['ymin']), int(result['ymax'])





    img_array = np.array(img)


    # crop to bounding box
    img_array = img_array[ymin:ymax, xmin:xmax, :]

    # make greyscale
    img_array = img_array.mean(axis = 2, keepdims = True)

    # expand dim to have (m, n, 1) shape
    
    

    
    step_size_i = img_array.shape[0] // 28
    step_size_j = img_array.shape[1] // 28

    img_array = skimage.measure.block_reduce(img_array, (step_size_i, step_size_j, 1), np.mean)


    #if the shape is not divisible by 28, we have dimensions of size 29
    img_array = img_array[:28, :28, :]

    # to send to model, we need the batch dimension as well
    img_array = np.expand_dims(img_array, 0)

    
    #shows pixelated image
    #st.pyplot(plt.imshow(img_array.reshape(28,28,-1), cmap='gray').figure)
    img_array = img_array/255

    index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}


    
    prediction = np.argmax(klaasmodel.predict(img_array), axis = 1 )[0]
    pred_letter = index_to_letter[prediction]
    st.session_state.count.append(pred_letter)


def main():
    #page header

    st.sidebar.image("ASL.png", use_column_width=True)

    #textlist = []
    #st.session_state['textlist'] = []


    st.markdown('<p style="font-size:60px;">H.A.L. 9000 Sign Language Translator</p>', unsafe_allow_html=True)
    
    
    #st.markdown(torch.__version__, torchvision.__version__)


    #bring in models
    #st.markdown("before model import")
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
    klaasmodel = tf.keras.models.load_model('../model_file/sign_language_model.keras')
    #st.markdown("after model import")

    #remove runs folder just in case 

    # if 'count' not in st.session_state:
    #     st.session_state.count = 0

    # increment = st.button('Increment')
    # if increment:
    #     st.session_state.count += 1

    if 'count' in st.session_state:

        #st.markdown(st.session_state.count)
        st.markdown(f'<p style="font-size:40px;">{"".join(st.session_state.count)}</p>', unsafe_allow_html=True)
   
    imageCaptured = st.camera_input("Do some sign language!" , help ="Do some sign language!" )

    fancybutton = st.button('Translate', on_click = increment , args = (imageCaptured,))
   



    
    #if imageCaptured is not None:
        
        
    #     img = Image.open(imageCaptured)

    #     os.system('rm -rf runs')

    #     results = model_yolo(img)
    #     results.save()


    #     st.image('runs/detect/exp/image0.jpg' , width = 300)

    #     result = results.pandas().xyxy[0]
    #     xmin, xmax, ymin, ymax = int(result['xmin']), int(result['xmax']), int(result['ymin']), int(result['ymax'])





    #     img_array = np.array(img)


    #     # crop to bounding box
    #     img_array = img_array[ymin:ymax, xmin:xmax, :]

    #     # make greyscale
    #     img_array = img_array.mean(axis = 2, keepdims = True)

    #     # expand dim to have (m, n, 1) shape
        
        

        
    #     step_size_i = img_array.shape[0] // 28
    #     step_size_j = img_array.shape[1] // 28

    #     img_array = skimage.measure.block_reduce(img_array, (step_size_i, step_size_j, 1), np.mean)


    #     #if the shape is not divisible by 28, we have dimensions of size 29
    #     img_array = img_array[:28, :28, :]

    #     # to send to model, we need the batch dimension as well
    #     img_array = np.expand_dims(img_array, 0)

       

    #     st.pyplot(plt.imshow(img_array.reshape(28,28,-1), cmap='gray').figure)
    #     img_array = img_array/255

    #     index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}


        
    #     prediction = np.argmax(klaasmodel.predict(img_array), axis = 1 )[0]
    #     pred_letter = index_to_letter[prediction]
    #     #textlist.append(pred_letter)
    #     st.session_state['textlist'] += 1



    #     #st.markdown(textlist)
    #     #st.markdown(pred_letter)
    #     st.markdown(st.session_state['textlist'])

    #st.write(textlist)
























        






   










if __name__ == '__main__':
    main()









