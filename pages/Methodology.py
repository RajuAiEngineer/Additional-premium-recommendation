import pandas as pd
import streamlit as st

# Set page title
st.title("Methodology")

st.image('./data/Methodology.png')

HD_csv_file_path = './data/healthdeclaration.csv'
APMatrix_csv_file_path = './data/APMatrix.csv'

# Read the CSV file with specified encoding
df = pd.read_csv(HD_csv_file_path, encoding='latin1')

# Create a download button for the first file
st.download_button(
    label="Download Health Declaration data CSV file",
    data=df.to_csv(index=False).encode('latin-1'),
    file_name="healthdeclaration.csv",  # Add .csv extension
    mime="text/csv"
)

# Read the second CSV file with specified encoding
df2 = pd.read_csv(APMatrix_csv_file_path, encoding='latin1')

# Create a download button for the second file
st.download_button(
    label="Download AP matrix data CSV file",
    data=df2.to_csv(index=False).encode('latin-1'),
    file_name="APMatrix.csv",  # Add .csv extension
    mime="text/csv"
)
# Footer
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
st.write("Thank you for choosing our application. We're excited to bring precision and ease to the process of illness classification and premium prediction.")

st.caption("Powered by EDD | Bringing AI-driven solutions")
