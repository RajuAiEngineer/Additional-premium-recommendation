import streamlit as st

# Set page title
st.title("About Us")

# Project Title
st.header("AI-Powered Recommendation Engine Using Embeddings")

# Problem Statement
st.markdown("""
CPFB uses Pega Case Management to automate processing. There are many workflow can’t be automated because the process requires staff to read long text then determine important keywords 
e.g. Staff need to read health declaration submitted by Member then decide whether an additional premium is required. 

If there is a recommendation engine that can classify, we can then integrate with Pega and form end-to-end processing.
""")

# Objective
st.markdown("""
**Proof of concept:** to learn how to  use Embeddings to make recommendation, and whether the model is accurate enough for real application.
For the real application, this function will be an API i.e. Pega call API to pass bunch of text, and the API will returns Illness type and whether Addtional Premium is required

""")

# Project Scope
st.markdown("""
1. Generate two models 
   a. Convert health declaration text into Illness type/name, up to five 
   b. Combine the derived illness type, and recommend whether additional premium is required

2. Build UI to enable user to get recommendations

""")


# Data sources
st.markdown("""
We use dummy data. 

There are two CSV used - you can download from Methodology page
1. Health declaration text and Illness Type
2. Combination of Illness type and whether Additional Premium is required
""")


# Project Features
st.markdown("""
1. UI to get recommendation
2. Function to update the two models

""")



# Footer
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
st.write("Thank you for choosing the our application. We're excited to bring precision and ease to the process of illness classification and premium pediction.")

st.caption("Powered by EDD | Bringing AI-driven solutions")