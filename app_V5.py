
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

st.set_page_config(layout="centered")
st.session_state['textlist'] = 0




def increment(imageCaptured):
    if 'count' not in st.session_state:
        st.session_state.count = []

    
    #st.session_state.count.append('test')
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
    klaasmodel = tf.keras.models.load_model('./model_file/sign_language_model.keras')

    img = Image.open(imageCaptured)

    os.system('rm -rf runs')

    results = model_yolo(img)
    results.save()


    #st.image('runs/detect/exp/image0.jpg' , width = 300)

    # result = results.pandas().xyxy[0]
    # xmin, xmax, ymin, ymax = int(result['xmin']), int(result['xmax']), int(result['ymin']), int(result['ymax'])

    # store the bounding box for the hand, as detected by the YOLO model
    try:
        # if mulitple boxes, take the largest one
        result = results.pandas().xyxy[0]
        result['area'] = (result['xmax'] - result['xmin']) * (result['ymax'] - result['ymin'])
        result = result.sort_values(by = ['area'], ascending = False)
        result = result.iloc[0]
        xmin, xmax, ymin, ymax = int(result['xmin']), int(result['xmax']), int(result['ymin']), int(result['ymax'])
    except:
        # print(f'Result not found for letter {value}, using entire image')
        xmin, xmax, ymin, ymax = 0, img.shape[1], 0, img.shape[0]
    # xmin, xmax, ymin, ymax = 0, hand.shape[1], 0, hand.shape[0]



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


    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
    klaasmodel = tf.keras.models.load_model('./model_file/sign_language_model.keras')

    tab1, tab2 = st.tabs(["Camera Translator", "Picture Translator"])


    with tab1:
        #page header

        st.sidebar.image("gulogo.png" , use_column_width = True)
        st.sidebar.image("ASL.png", use_column_width = True)
        st.sidebar.markdown('<br>', unsafe_allow_html = True)
        st.sidebar.markdown('<center><b><p style="font-size:20px;">Group 1</p></b></center>', unsafe_allow_html = True)
        st.sidebar.markdown('<center><p>Brendan Baker, Justin Ceresa</p><p>Klaas van Kempen, Matt Moriarty</p></center>', unsafe_allow_html = True)

        #textlist = []
        #st.session_state['textlist'] = []


        #st.markdown('<p style="font-size:60px;">Sign Language Translator</p>', unsafe_allow_html=True)
        st.markdown('<center><p style="font-size:60px; color:#ffffff; background-color:#041E42; padding: 10px;">Sign Language Translator</p></center>', unsafe_allow_html=True)
        
        
        #st.markdown(torch.__version__, torchvision.__version__)


        #bring in models
        #st.markdown("before model import")
        #model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        #klaasmodel = tf.keras.models.load_model('../model_file/sign_language_model.keras')
        #st.markdown("after model import")

        #remove runs folder just in case 

        # if 'count' not in st.session_state:
        #     st.session_state.count = 0

        # increment = st.button('Increment')
        # if increment:
        #     st.session_state.count += 1

        if 'count' in st.session_state:

            #st.markdown(st.session_state.count)
            st.markdown(f'<p style="font-size:40px;">Translated Letters : {"".join(st.session_state.count)}</p>', unsafe_allow_html=True)
    
        imageCaptured = st.camera_input("Do some sign language!" , help ="Do some sign language!" )

        fancybutton = st.button('Translate', use_container_width = True, on_click = increment , args = (imageCaptured,))



    with tab2:

        # test = st.file_uploader("input the sign_mnist_test.csv file here please!!")
        test = pd.read_csv('./sign_mnist_test.csv')

        # klass data prep
        test["label"][test["label"]>=10] = test["label"] -1

        test_data = test.drop("label", axis = 1)
        test_labels = test['label']
        test_data = np.array(test_data)

        test_data = test_data/255

        test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
        
        index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}
        letter_to_index = dict([(value, key) for key, value in index_to_letter.items()])

        if test_data is not None:
            # user picks a letter
            letter = st.text_input('Letter Input', '', placeholder = 'Enter a letter...')

            if letter != '':
                index = letter_to_index[letter]

                # filter test data for the chosen letter
                df = test_data[test_labels == index, :]

                # obtain one random row of the chosen letter
                image_num = np.random.randint(1, df.shape[0])
                df = df[[image_num], :]

                # make prediction and retrieve predicted letter
                predictions = klaasmodel.predict(df, verbose = 0)
                predicted_label = np.argmax(predictions, axis = 1)[0]

                pred_letter = index_to_letter[predicted_label]
                true_letter = letter

                ## note the changes i made to this line.  I need to add the st.pyplot and the .figure here at the end
                fig, ax = plt.subplots(1, 1, figsize = (5, 5))
                plt.imshow(df.reshape(28, 28, 1), cmap = 'gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('letter_image.png')

                col1, col2, col3 = st.columns([1, 3, 1])

                with col1:
                    st.write(' ')

                with col2:
                    st.image('letter_image.png')

                with col3:
                    st.write(' ')
                # st.pyplot(fig.figure)

                # st.pyplot(plt.imshow(df.reshape(28,28,1), cmap='gray').figure)
                # plt.axis('off')
                plt.show()

                df_table = pd.DataFrame({'Predicted Letter': [pred_letter],
                                         'Actual Letter': [true_letter]})
                df_table.set_index(df_table.columns[0])
                
                st.table(df_table)

                #st.markdown(pred_letter)
                # st.markdown(f'<p style="font-size:40px; padding: 10px;">Predicted Letter : {pred_letter}</p>', unsafe_allow_html=True)
                #st.markdown( true_letter)
                # st.markdown(f'<p style="font-size:40px; padding: 10px;">Actual Letter : {true_letter}</p>', unsafe_allow_html=True)



        # if test is not None:
        #     test = pd.read_csv(test)

        #     # klass data prep
        #     test["label"][test["label"]>=10] = test["label"] -1

        #     test_data = test.drop("label", axis = 1)
        #     test_labels = test['label']
        #     test_data = np.array(test_data)

        #     test_data = test_data/255

        #     test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

        #     #st.markdown("testtesttest")

        #     # mapping function for label index to true alphabetical character
        #     index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's', 18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}

        #     predictions = klaasmodel.predict(test_data, verbose = 0)
        #     predicted_labels = np.argmax(predictions, axis=1)

        #     # random test image
        #     randomnum = np.random.randint(1,7170)
        #     image_num = randomnum

        #     ### note the changes i made to this line.  I need to add the st.pyplot and the .figure here at the end
        #     st.pyplot(plt.imshow(test.iloc[image_num][1:].values.reshape(28,28,1), cmap='gray').figure)
        #     plt.show()

        #     # reshape and normalize test image
        #     img = np.array(test.iloc[image_num][1:])
        #     img = img.reshape(1,28,28,1)
        #     img = img/255

        #     # make prediction on test image
        #     prediction = np.argmax(klaasmodel.predict(img), axis = 1 )[0]
        #     pred_letter = index_to_letter[prediction]
        #     # return true prediction
        #     true_letter = index_to_letter[test.iloc[image_num][0]]

        #     #st.markdown(pred_letter)
        #     st.markdown(f'<p style="font-size:60px; padding: 10px;">Predicted Letter : {pred_letter}</p>', unsafe_allow_html=True)
        #     #st.markdown( true_letter)
        #     st.markdown(f'<p style="font-size:60px; padding: 10px;">Actual Letter : {true_letter}</p>', unsafe_allow_html=True)
            

   



   





















        






   










if __name__ == '__main__':
    main()









