import os
from openai import OpenAI
import streamlit as st
import tiktoken
import pandas as pd
import joblib
import json
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


if load_dotenv('.env'):
    # forlocal development
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
    OPENAI_KEY = st.secrets['OPENAI_API_KEY']


# Load trained models
illness_recommendation_Model = joblib.load('./data/healthdeclaration_Model.joblib')
premium_recommendation_model = joblib.load('./data/premiumMatrix_Model.joblib')

# Pass the API Key to the OpenAI Client
client = OpenAI(api_key=OPENAI_KEY)

# Function to get embedding from OpenAI API
def get_embedding(input, model='text-embedding-3-small'):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return [x.embedding for x in response.data]


# Function to generate illness type embeddings
def generate_illness_embeddings(file_path, output_path,progress_bar):
    try:
        # Load the original CSV file
        data = pd.read_csv(file_path, sep=',', encoding='latin-1')
        # Combine illness columns if they're not already combined
        if 'combined_illnesses' not in data.columns:
            data['combined_illnesses'] = data['declaration'] + ' ' + data['illness type']
        embeddings = []
        total_declarations = len(data['combined_illnesses'])

        # Iterate through the declarations with a progress update
        for index, declaration in enumerate(data['combined_illnesses']):
            response = client.embeddings.create(input=declaration,model='text-embedding-3-small')
            # Extract and append embeddings
            embeddings.append(response.data[0].embedding)
            # Update the progress bar
            progress_percentage = (index + 1) / total_declarations
            progress_bar.progress(progress_percentage)
        # Convert embeddings list to numpy array and then to DataFrame
        embeddings_df = pd.DataFrame(
            np.array(embeddings), columns=[f'embedding_{i}' for i in range(1,1537)]
            )
        # Concatenate original data with embeddings
        result = pd.concat([data, embeddings_df], axis=1)
        # Save the result to a new CSV file
        result.to_csv(output_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# Function to generate Premium Matrix embeddings
def generate_AP_matrix_embeddings(file_path, output_path,progress_bar):
    try:
        # Load the original CSV file
        premium_data = pd.read_csv(file_path, sep=',', encoding='latin-1')
        # Combine illness columns if they're not already combined
        if 'combined_illnesses' not in premium_data.columns:
            premium_data['combined_illnesses'] = premium_data['Illness 1'] + ' ' + premium_data['Illness 2'] + ' ' + premium_data['Illness 3'] + ' ' + premium_data['Illness 4'] + ' ' + premium_data['Illness 5']
        premium_embeddings = []
        total_premium_data = len(premium_data['combined_illnesses'])

        # Iterate through the Premium Matrix with a progress update
        for index, illness in enumerate(premium_data['combined_illnesses']):
            response = client.embeddings.create(input=illness,model='text-embedding-3-small')
            # Extract and append embeddings
            premium_embeddings.append(response.data[0].embedding)
            # Update the progress bar
            progress_percentage = (index + 1) / total_premium_data
            progress_bar.progress(progress_percentage)
        # Convert embeddings list to numpy array and then to DataFrame
        premium_embeddings_df = pd.DataFrame(
            np.array(premium_embeddings), columns=[f'embedding_{i}' for i in range(1,1537)]
            )
        # Concatenate original data with embeddings
        premium_result = pd.concat([premium_data, premium_embeddings_df], axis=1)
        # Save the result to a new CSV file
        premium_result.to_csv(output_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# Model generation - healthdeclaration   
def generate_healthdeclaration_Model(file_path, output_path,progress_bar):
    try:
        # Model generation
        # Load data
        progress_percentage = 0
        data = pd.read_csv(file_path)
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Includes columns
        embedding_columns = [f'embedding_{i}' for i in range(1, 1537)]
        X = data[embedding_columns]
        y = data['illness type']
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Define parameter grid for GridSearchCV
        param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)
        # Perform Grid Search
        print("Performing grid search...")
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        progress_percentage += 40 
        progress_bar.progress(progress_percentage)
        # Get best model
        best_rf = grid_search.best_estimator_
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Make predictions
        y_pred = best_rf.predict(X_test)
        progress_percentage += 5
        progress_bar.progress(progress_percentage)
        # Save the model
        joblib.dump(best_rf, output_path)        
        
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


# Model generation - AP Matrix   
def generate_premiumMatrix_Model(file_path, output_path,progress_bar):
    try:
        # Model generation
        progress_percentage = 0
        # Load CSV with Premium Matrix embeddings
        premium_data = pd.read_csv(file_path)
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Includes columns
        embedding_columns = [f'embedding_{i}' for i in range(1, 1537)]
        X = premium_data[embedding_columns]
        y = premium_data['Need premium flag']
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Define parameter grid for GridSearchCV
        param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }
        
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        
        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)
        # Perform Grid Search
        print("Performing grid search...")
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        progress_percentage += 40 
        progress_bar.progress(progress_percentage)
        # Get best model
        best_rf = grid_search.best_estimator_
        progress_percentage += 10
        progress_bar.progress(progress_percentage)
        # Make predictions
        y_pred = best_rf.predict(X_test)
        progress_percentage += 5
        progress_bar.progress(progress_percentage)
        # Save the model
        joblib.dump(best_rf, output_path)        
        
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False


# Function to predict the illness type
def predict_illness(declaration, model=illness_recommendation_Model, embedding_model='text-embedding-3-small'):
    """Predicts the illness type and probability based on the provided declaration.

    Args:
      declaration: The declaration text describing the symptoms or condition.
      model: The trained Random Forest model for classification.
      embedding_model: The OpenAI model to use for generating embeddings.

    Returns:
      A tuple of the predicted illness type and its probability.
    """

    # Generate embedding for the declaration
    embedding = get_embedding(declaration, model=embedding_model)[0]

    # Convert the embedding to a DataFrame
    embedding_df = pd.DataFrame(
        [embedding], columns=[f'embedding_{i}' for i in range(1, 1537)]
    )

    # Make prediction using the loaded model
    predicted_illness = model.predict(embedding_df)[0]

    # Get the probability of the predicted illness
    illness_prob = model.predict_proba(embedding_df).max()

    return predicted_illness, illness_prob


# Example usage:
# user_input = input("Enter your health declaration: ")
# predicted_illness = predict_illness(user_input)
# print(f"Predicted Illness: {predicted_illness}")


# Function to predict whether the additional premium is needed or not
def predict_premium(illness_combination, model=premium_recommendation_model, embedding_model='text-embedding-3-small'):
    """Predicts if additional premium is needed based on the provided illness combination.

    Args:
      illness_combination: The concatenated string of all illnesses.
      model: The trained Random Forest model for classification.
      embedding_model: The OpenAI model to use for generating embeddings.

    Returns:
      A tuple of the prediction (Yes or No) if premium is needed, and the probability.
    """

    # Generate embedding for the illness combination
    embedding = get_embedding(illness_combination, model=embedding_model)[0]

    # Convert the embedding to a DataFrame
    embedding_df = pd.DataFrame(
        [embedding], columns=[f'embedding_{i}' for i in range(1, 1537)]
    )

    # Make prediction using the loaded model
    premium_recommendation = model.predict(embedding_df)[0]

    # Get the probability of the prediction
    premium_prob = model.predict_proba(embedding_df).max()

    return premium_recommendation, premium_prob

# Example usage:
# user_input = input("Enter your health declaration: ")
# premium_recommendation = predict_premium(user_input)
# print(f"Premium needed?: {premium_recommendation}")

# Function to load discussions from a JSON file


def load_discussions(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Function to save discussions to a JSON file


def save_discussions(file_path, discussions):
    with open(file_path, 'w') as f:
        json.dump(discussions, f)
