# Set up and run this Streamlit App
from ast import If
import streamlit as st
import time
from helper_functions import llm
from helper_functions.utility import check_password
#ssmb
# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="wide",
    page_title="My Streamlit App"
)

st.header("Illness and Additional Premium Recommendation Engine")
# endregion <--------- Streamlit App Configuration --------->

# Check if the password is correct.
if not check_password():
    st.stop()

st.subheader("Health Declaration to Illness Type", divider="gray")

# Initialize a variable to store all selected illnesses
ccIllness = ""

# Initialize session state for managing pages and inputs
if 'page' not in st.session_state:
    st.session_state.page = 'input'  # Start on the input page

# Initialize the session state variable if it doesn't exist
if "embedded_data" not in st.session_state:
    # or initialize it with some default value
    st.session_state.embedded_data = ['' for _ in range(5)]

# Initialize input fields in session state if they don't exist
if 'input_data' not in st.session_state:
    # Create a list for 5 inputs
    st.session_state.input_data = ['' for _ in range(5)]

# Initialize concat data fields in session state if they don't exist
if 'concat_data' not in st.session_state:
    st.session_state.concat_data = ""

# Initialize Combined_data fields in session state if they don't exist
if 'Combined_data' not in st.session_state:
    st.session_state.Combined_data = ""

# Initialize AP_Needed data fields in session state if they don't exist
if 'AP_Needed' not in st.session_state:
    st.session_state.AP_Needed = ""

# Initialize AP_Prob data fields in session state if they don't exist
if 'AP_Prob' not in st.session_state:
    st.session_state.AP_Prob = ""
Pre_prob = 0.0

# Page 1: Input Page

if st.session_state.page == 'input':
    st.markdown(""" **Instruction:**""")
    st.markdown("""
    1) Enter up to five health declaration texts and click **Submit**.
    2) Take note of the probability. User should not use recommendation that has lower than **0.1** probability.
    """)
    # Maincol1, Maincol2 = st.columns(
    #     [5, 2], vertical_alignment="center")  # first table
    # with Maincol1:

    # Column headers 2nd table
    # Adjust column widths for a tighter layout
    col1, col2, col3, col4 = st.columns([0.3, 5, 1, 0.7])
    with col1:
        # Header for row numbers
        st.markdown('**<span>No</span>**', unsafe_allow_html=True)
    with col2:
        st.markdown('**<span>Declaration</span>**',
                    unsafe_allow_html=True)  # Header for Column 1
    with col3:
        st.markdown('**<span>Illness Type</span>**',
                    unsafe_allow_html=True)
    with col4:
        st.markdown(
            '**<span>Probability</span>**', unsafe_allow_html=True)

    # Create rows with text inputs and row numbers in a compact format
    for i in range(5):
        # Adjust columns to remove extra spacing
        col1, col2, col3, col4 = st.columns(
            [0.3, 5, 1, 0.7], vertical_alignment="center", gap="small")
        with col1:
            st.write(f"{i + 1}")  # Display row number
        with col2:
            st.session_state.input_data[i] = st.text_area(
                "Declaration",
                key=f"text1_{i}",
                label_visibility="collapsed",
                # Retain the previous value
                value=st.session_state.input_data[i]
            )  # Store input in session state

        # Update to include probability
        with col3:
            # selectedIllness = st.session_state.embedded_data[i]
            # st.write(f"{selectedIllness}")
            st.write(st.session_state.embedded_data[i])

        with col4:
            # Add a placeholder for probability
            probability = st.session_state.get(f"probability_{i}", "")
            # Apply color based on probability threshold
            if probability:
                if float(probability) < 0.1:
                    # Display in red for low probability
                    st.markdown(f"<span style='color:red;'>{
                                probability}</span>", unsafe_allow_html=True)
                else:
                    # Display in green for acceptable probability
                    st.markdown(f"<span style='color:green;'>{
                                probability}</span>", unsafe_allow_html=True)

    # add section divider
    st.subheader("Additional Premium Recommendation", divider="grey")

    container = st.container(border=True)
    container.write(f"Combined Illness Types : {
                    st.session_state.Combined_data}")

    container.write("**Additional premium required?**")
    if st.session_state.AP_Prob:
        Pre_prob = float(st.session_state.AP_Prob)
        if float(Pre_prob) < 0.1:
            container.warning(
                f"NA - <0.1 probability{st.session_state.AP_Prob}")
        elif st.session_state.AP_Needed in "No":
            container.success(f"{st.session_state.AP_Needed}")
        else:
            container.error(f"{st.session_state.AP_Needed}")

        if float(Pre_prob) < 0.1:
            # Display in red for low probability
            container.markdown(
                f"**Probability :** <span style='color:red;'>{Pre_prob}</span>", unsafe_allow_html=True)
        else:
            # Display in green for acceptable probability
            container.markdown(
                f"**Probability :** <span style='color:green;'>{Pre_prob}</span>", unsafe_allow_html=True)

    # Submit button and clear button
    btncol1, btncol2, btncol3 = st.columns([1, 8, 1])

    with btncol1:
        # Submit button logic update
        if st.button("Submit"):
            st.session_state.concat_data = ""
            st.session_state.Combined_data = ""
            # Predict illness and probability
            for i in range(5):
                if st.session_state.input_data[i]:
                    # Predict illness and probability
                    illness, prob = llm.predict_illness(
                        st.session_state.input_data[i])

                    # Store illness and probability in session state
                    st.session_state.embedded_data[i] = illness
                    # Format probability to 2 decimal places
                    st.session_state[f"probability_{i}"] = f"{prob:.2f}"
                    selectedIllness = illness
                    # Concatenate selected illnesses into CCvalue
                    st.session_state.concat_data += f"{selectedIllness} "
                    if st.session_state.Combined_data:
                        st.session_state.Combined_data += f"+ {
                            selectedIllness} "
                    else:
                        st.session_state.Combined_data += f"{selectedIllness} "

                    ccIllness = st.session_state.concat_data
                else:
                    st.session_state.embedded_data[i] = ""
                    st.session_state[f"probability_{i}"] = ""

            # Predict additional premium if any illness is present
            if ccIllness:
                # Predict additional premium and its probability
                ap_needed, ap_prob = llm.predict_premium(ccIllness)

                # Store AP_Needed with probability
                st.session_state.AP_Needed = f"{ap_needed}"
                st.session_state.AP_Prob = f"{ap_prob:.2f}"
                st.rerun()
            else:
                # Toast notification (temporary pop-up style)
                st.toast("Please enter declaration before submitting!", icon="📢")

    with btncol3:
        # Button to clear session state
        if st.button("Clear"):
            st.session_state.clear()
            st.rerun()

# Footer
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
# st.write("Thank you for choosing the our application. We're excited to bring precision and ease to the process of illness classification and premium pediction.")

# Create an expander for extra details
with st.expander("""

**IMPORTANT NOTICE:** This web application is developed as a proof-of-concept prototype. The information provided here is **NOT intended for actual** usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

**Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.**

Always consult with qualified professionals for accurate and personalized advice."""):
    st.write("")
st.caption("Powered by EDD | Bringing AI-driven solutions")
