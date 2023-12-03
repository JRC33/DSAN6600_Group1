
### Package Imports ###

# streamlit library
import streamlit as st

# data manipulation libraries
import pandas as pd
import numpy as np

# deep learning libraries
import tensorflow as tf
from tensorflow import keras
import torch
import torchvision

# os-related libraries
import os
import shutil

# plotting/extra libraries
import matplotlib.pyplot as plt
import cv2
import skimage.measure
import IPython
import PIL
from PIL import Image

# center the layout of the page
st.set_page_config(layout = "centered")

# load the YOLO object detection model and our own sign language translation CNN model
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
klaasmodel = tf.keras.models.load_model('./model_file/sign_language_model.keras')

# store mapping from predicted indices to translated letters and back
index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
                   6: 'g', 7: 'h', 8: 'i', 9: 'k', 10: 'l', 11: 'm',
                   12: 'n', 13: 'o', 14: 'p', 15: 'q', 16: 'r', 17: 's',
                   18: 't', 19: 'u', 20: 'v', 21: 'w', 22: 'x', 23: 'y'}
letter_to_index = dict([(value, key) for key, value in index_to_letter.items()])

# store a 'session state' variable, which persists through repeated actions in the application
st.session_state['count'] = 0



# define a callback function to translate an image, triggered when a button is clicked
def translate_image(imageCaptured):

    # store a 'session state' variable, which persists through repeated actions in the application
    if 'count' not in st.session_state:
        st.session_state.count = []

    # open the captured image
    img = Image.open(imageCaptured)

    # send the image through the YOLO model
    results = model_yolo(img)
    results.save()

    # store the bounding box for the hand, as detected by the YOLO model
    try:
        # start by obtaining the results as a pandas dataframe
        result = results.pandas().xyxy[0]
        # calculate bounding box areas and take the largest detected bounding box in case there are multiple
        result['area'] = (result['xmax'] - result['xmin']) * (result['ymax'] - result['ymin'])
        result = result.sort_values(by = ['area'], ascending = False)
        result = result.iloc[0]
        # store the bounds of the bounding box
        xmin, xmax, ymin, ymax = int(result['xmin']), int(result['xmax']), int(result['ymin']), int(result['ymax'])
    except:
        # if there are somehow no bounding boxes detected, use the whole image instead of cropping
        xmin, xmax, ymin, ymax = 0, img.shape[1], 0, img.shape[0]

    # store the image as a numpy array for easy manipulation
    img_array = np.array(img)

    # crop the image to the detected bounding box
    # remember rows are actually y-values and columns are x-values
    img_array = img_array[ymin:ymax, xmin:xmax, :]

    # make image greyscale by averaging the color dimension ( shape (m, n, 1) )
    img_array = img_array.mean(axis = 2, keepdims = True)

    # define step sizes for reducing the image to shape (28, 28, 1)
    step_size_i = img_array.shape[0] // 28
    step_size_j = img_array.shape[1] // 28

    # scan over the image, averaging out blocks of pixels
    img_array = skimage.measure.block_reduce(img_array, (step_size_i, step_size_j, 1), np.mean)

    # if the previous shape is not divisible by 28, we have dimensions of size 29,
    # so we need to get rid of the excess row and column (size equal to remainder)
    img_array = img_array[:28, :28, :]

    # to send to model, we need the batch dimension as well ( now shape (1, 28, 28, 1) )
    img_array = np.expand_dims(img_array, 0)

    # finally, normalize pixel intensities from [0, 255] to [0, 1]
    img_array = img_array / 255

    # use the sign language translation CNN model to translate the image and store it in the persistent session state
    prediction = np.argmax(klaasmodel.predict(img_array), axis = 1 )[0]
    pred_letter = index_to_letter[prediction]
    st.session_state.count.append(pred_letter)


# main streamlit function
def main():

    # create two tabs, one for live camera sign translation, one for translating images from the dataset
    tab1, tab2 = st.tabs(["Camera Translator", "Picture Translator"])


    # define the functionality of tab 1
    with tab1:

        # include Georgetown University logo and a sign language image
        st.sidebar.image("gulogo.png" , use_column_width = True)
        st.sidebar.image("ASL.png", use_column_width = True)

        # include names and group number
        st.sidebar.markdown('<br>', unsafe_allow_html = True)
        st.sidebar.markdown('<center><b><p style="font-size:20px;">Group 1</p></b></center>', unsafe_allow_html = True)
        st.sidebar.markdown('<center><p>Brendan Baker, Justin Ceresa</p><p>Klaas van Kempen, Matt Moriarty</p></center>', unsafe_allow_html = True)

        # create a large title to indicate that this is a sign language translator
        st.markdown('<center><p style="font-size:60px; color:#ffffff; background-color:#041E42; padding: 10px;">Sign Language Translator</p></center>', unsafe_allow_html=True)
        
        # only display the translated letters if we've attempted to translate at least one
        if 'count' in st.session_state:

            # display the translated letters consecutively
            st.markdown(f'<p style="font-size:40px;">Translated Letters : {"".join(st.session_state.count)}</p>', unsafe_allow_html=True)

        # capture a live image using the camera
        imageCaptured = st.camera_input("Do some sign language!" , help = "Do some sign language!")

        # create a translation button that kicks off the deep learning pipeline after collecting an image
        # when clicked, this button triggers the 'translate_image' callback function with arguments 'args'
        translate_button = st.button('Translate', use_container_width = True, on_click = translate_image, args = (imageCaptured,))


    # define the functionality of tab 1
    with tab2:

        # read in the "clean" pre-prepared sign language data for use in this tab
        test = pd.read_csv('./sign_mnist_test.csv')

        # make the labels of the data consecutive since we don't have 'J' or 'Z'
        test["label"][test["label"] >= 10] = test["label"] - 1

        # split the data into the pixel components (X) and the labels (Y)
        test_data = test.drop("label", axis = 1)
        test_labels = test['label']

        # normalize the data using numpy from [0, 255] to [0, 1]
        test_data = np.array(test_data)
        test_data = test_data/255

        # introduce the batch dimension, as the model expects
        test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)        

        # if we've loaded in the data properly, let the user select a letter
        if test_data is not None:
            
            # store the user's letter input
            letter = st.text_input('Letter Input', '', placeholder = 'Enter a letter...')

            # perform the task if the user typed in one letter
            if len(letter) == 1:

                # filter test data for the chosen letter
                index = letter_to_index[letter]
                df = test_data[test_labels == index, :]

                # obtain one random row of the chosen letter
                image_num = np.random.randint(1, df.shape[0])
                df = df[[image_num], :]

                # make prediction and retrieve predicted letter
                predictions = klaasmodel.predict(df, verbose = 0)
                predicted_label = np.argmax(predictions, axis = 1)[0]

                # store the predicted letter and the true letter
                pred_letter = index_to_letter[predicted_label]
                true_letter = letter

                # save an image displaying the randomly selected image
                fig, ax = plt.subplots(1, 1, figsize = (5, 5))
                plt.imshow(df.reshape(28, 28, 1), cmap = 'gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('letter_image.png')

                # use columns for the purpose of centering the image and making it larger
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(' ')
                with col2:
                    st.image('letter_image.png')
                with col3:
                    st.write(' ')

                # st.pyplot(plt.imshow(df.reshape(28,28,1), cmap='gray').figure)
                # plt.axis('off')
                # plt.show()

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









