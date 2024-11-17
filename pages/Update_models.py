import streamlit as st
import os
from helper_functions import utility
from helper_functions import llm
from helper_functions.utility import check_password

# Page tittle
st.title("Update Models Page")

# Check if the password is correct.
if not check_password():
    st.stop()

# Declaration Section
st.subheader("Illness Type Recommendation",divider="gray")
st.markdown(""" **Instructions:**""")
st.markdown("""
    1) Upload file.
    2) Update model.
    """)

# Set file path
declaration_file_path = "./data/healthdeclaration.csv"
declaration_file_path_Embeddings = "./data/healthdeclaration_embeddings.csv"
declaration_model_path = "./data/healthdeclaration_Model.joblib"

declaration_file = st.file_uploader("Upload Helath Declaration CSV File", type=["csv"])

if st.button("Upload file"):
    if declaration_file is not None:
        if utility.save_uploaded_file(declaration_file, declaration_file_path):
            st.success("File uploaded successfully!")
        else:
            st.error("Failed to upload file.")
    else:
        st.warning("Please upload a file.")

# if st.button("Generate 
#Declaration Embedding"):
#     progress_bar1 = st.progress(0)
#     if llm.generate_illness_embeddings(declaration_file_path, declaration_file_path_Embeddings,progress_bar1):
#         progress_bar1.progress(100)
#         st.success("Embeddings generated successfully!")
#     else:
#         st.error("Failed to generate embeddings.")

if st.button("Update model"):
    progress_text1 = "Embedding operation in progress. Please wait."
    progress_text = "Model upate operation in progress. Please wait."
    progress_bar1 = st.progress(0,text=progress_text1)
    progress_bar = st.progress(0,text=progress_text)
    if llm.generate_illness_embeddings(declaration_file_path, declaration_file_path_Embeddings,progress_bar1):
        progress_bar1.progress(100)
        st.success("Embeddings generated successfully!")
        
        if llm.generate_healthdeclaration_Model(declaration_file_path_Embeddings, declaration_model_path,progress_bar):
            progress_bar.progress(100)
            st.success("Model generated successfully!")
        else:
            st.error("Failed to generate model.")
    else:
        st.error("Failed to generate embeddings.")

# Additional Premium Matrix Section
st.subheader("Additional Premium Recommendation",divider="gray")
st.markdown(""" **Instructions:**""")
st.markdown("""
    1) Upload file.
    2) Update model.
    """)
# Set file path
matrix_file_path = "./data/APMatrix.csv"
matrix_file_path_Embeddings = "./data/APMatrix_embeddings.csv"
matrix_model_path =  "./data/premiumMatrix_Model.joblib"

matrix_file = st.file_uploader("Upload Additional Premium Matrix CSV File", type=["csv"])
if st.button("Upload_file"):
    if matrix_file is not None:
        if utility.save_uploaded_file(matrix_file, matrix_file_path):
            st.success("File uploaded successfully!")
        else:
            st.error("Failed to upload file.")
    else:
        st.warning("Please upload a file.")

# if st.button("Generate AP Matrix Embeddings"):
#     progress_bar = st.progress(0)
#     if llm.generate_AP_matrix_embeddings(matrix_file_path, matrix_file_path_Embeddings,progress_bar):
#         progress_bar.progress(100)
#         st.success("Embeddings generated successfully!")
#     else:
#         st.error("Failed to generate embeddings.")

if st.button("Update_model"):
    progress_text1 = "Embedding operation in progress. Please wait."
    progress_text = "Model upate operation in progress. Please wait."
    progress_bar1 = st.progress(0,text=progress_text1)
    progress_bar = st.progress(0,text=progress_text)
    if llm.generate_AP_matrix_embeddings(matrix_file_path, matrix_file_path_Embeddings,progress_bar1):
        progress_bar1.progress(100)
        st.success("Embeddings generated successfully!")
        if llm.generate_premiumMatrix_Model(matrix_file_path_Embeddings, matrix_model_path,progress_bar):
            progress_bar.progress(100)
            st.success("Model generated successfully!")
        else:
            st.error("Failed to generate model.")
    else:
        st.error("Failed to generate embeddings.")

# Footer
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
st.write("Thank you for choosing our application. We're excited to bring precision and ease to the process of illness classification and premium prediction.")

st.caption("Powered by EDD | Bringing AI-driven solutions")
