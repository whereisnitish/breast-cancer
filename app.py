import numpy as np
import pickle
import pandas as pd

import streamlit as st 



pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_Breast_Cancer (texture_mean, compactness_mean, symmetry_mean, radius_se, compactness_se, radius_worst, perimeter_worst):
    
    
   
    prediction=classifier.predict([[texture_mean, compactness_mean, symmetry_mean, radius_se, compactness_se, radius_worst, perimeter_worst]])
    print(prediction)
    return prediction



def main():
    st.title("Breast Cancer Classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Breast Cancer Classification ML App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    texture_mean = st.text_input("texture_mean","Type Here")
    compactness_mean = st.text_input("compactness_mean","Type Here")
    symmetry_mean = st.text_input("symmetry_mean","Type Here")
    radius_se = st.text_input("radius_se","Type Here")
    compactness_se = st.text_input("compactness_se","Type Here")
    radius_worst = st.text_input("radius_worst","Type Here")
    perimeter_worst= st.text_input("perimeter_worst","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_Breast_Cancer (texture_mean, compactness_mean, symmetry_mean	,radius_se, compactness_se, radius_worst, perimeter_worst)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("M = malignant, B = benign")
        st.text("M = 1")
        st.text("B = 0")
        st.text("Tumors can be benign (noncancerous) or malignant (cancerous).Benign tumors tend to grow slowly and do not spread. Malignant tumors can grow rapidly, invade and destroy nearby normal tissues, and spread throughout the body.")
        st.text("Worldwide, breast cancer is the most common type of cancer in women and the second highest in terms of mortality rates.Diagnosis of breast cancer is performed when an abnormal lump is found (from self-examination or x-ray) or a tiny speck of calcium is seen (on an x-ray). After a suspicious lump is found, the doctor will conduct a diagnosis to determine whether it is cancerous and, if so, whether it has spread to other parts of the body.")
        

if __name__=='__main__':
    main()
    