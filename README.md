# DSAN6600 Group1

The goal of this project is to build a streamlit application which takes in live video of sign language and returns the predicted character associated with the current hand sign. The project consists of the following parts. 

**1. Data**
- This project uses a labeled dataset of hand signs. Each image is a hand sign and has a label indicating which letter is represents. All letters except for J and Z are represented in the dataset (J and Z require motion of the hand in their signs). Each letter is represented many times by different individuals in the dataset. The data can be found on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
  
**2. Data Preparation, Modeling, Evaluation**
- Using a jupyter notebook we prepared the input data and built a modeling pipeline to classify different signs. We also evaluated the performance of the model. The workflow can be found `/model_pipeline/pipeline`.
- The trained model is saved under `/model_file/sign_language_model.keras`

**3. Streamlit Application**
- `                 `

