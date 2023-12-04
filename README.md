# DSAN6600 Group1

- Brendan Baker
- Justin Ceresa
- Klaas van Kempen
- Matt Moriarty

The goal of this project is to build a streamlit application which takes in live video of sign language and returns the predicted character associated with the current hand sign. The project consists of the following parts. 

**1. Data**
- This project uses a labeled dataset of hand signs. Each image is a hand sign and has a label indicating which letter is represents. All letters except for J and Z are represented in the dataset (J and Z require motion of the hand in their signs). Each letter is represented many times by different individuals in the dataset. The data can be found on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
  
**2. Data Preparation, Modeling, Evaluation**
- Using a jupyter notebook we prepared the input data and built a modeling pipeline to classify different signs. We also evaluated the performance of the model. The workflow can be found in the `model_pipeline/` directory.
- The trained model is saved under `model_file/sign_language_model.keras`

**3. Streamlit Application**
- We culminated our analysis into a deployable [Streamlit](https://streamlit.io/) application, which can be found [here](https://dsan6600group1-z9ub9qpuzfqqjp6o4vbappy.streamlit.app/). The main driver of this application is the Python script that runs as the back end, which is called `app_V5.py`.

**4. Project Report**
- Our Project Report and other deliverables can be found in the `final_paper_files/` directory.