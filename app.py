import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dateutil import parser as date_parser

import pathlib

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Logic for data cleaning tasks
#Tasks-
#Missing value handling
#Maybe outlier handling(?)
#Formatting and standardization
#Date/Time manipulation
# Feature scaling and normalization
#Categorical encoding

# Function to handle Missing values
def handle_missing_values(df):
    total_samples = len(df)
    missing_values_present = False  # Flag to indicate if missing values were present
    
    for column in df.columns:
        missing_values_count = df[column].isnull().sum()
        missing_values_ratio = missing_values_count / total_samples
        
        if missing_values_count > 0:
            missing_values_present = True  # Set flag to True if missing values are found
            if df[column].dtype == 'object':
                # Categorical data handling
                if missing_values_ratio <= 0.05:
                    df[column].fillna(df[column].mode()[0], inplace=True)
                else:
                    classifier = RandomForestClassifier()
                    X_train = df.dropna(subset=[column])
                    y_train = X_train[column]
                    X_train = X_train.drop(columns=[column])
                    classifier.fit(X_train, y_train)
                    X_missing = df[df[column].isnull()].drop(columns=[column])
                    imputed_values = classifier.predict(X_missing)
                    df.loc[df[column].isnull(), column] = imputed_values
            else:
                # Numerical data handling
                if missing_values_ratio <= 0.05:
                    df.dropna(subset=[column], inplace=True)
                elif pd.Series(df[column]).is_unique:
                    df[column].fillna(df[column].median(), inplace=True)
                else:
                    if missing_values_ratio <= 0.2:
                        imputer = KNNImputer(n_neighbors=5)
                        df[column] = imputer.fit_transform(df[[column]])
                    else:
                        imputer = IterativeImputer(random_state=0)
                        df[column] = imputer.fit_transform(df[[column]])
    
    if missing_values_present:
        cleaning_status = "Missing values handled."
    else:
        cleaning_status = "No missing values found."
    
    return df, cleaning_status

#Function to handle outliers

def handle_outliers(df):
    outlier_handling_status = ""
    
    #Loop through numerical columns to handle outliers
    for column in df.select_dtypes(include=np.number).columns:
        #Outlier trimming
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1-1.5*iqr
        upper_bound = q3 +1.5*iqr
        df[column] = np.where(df[column]<lower_bound,lower_bound,df[column])
        df[column] = np.where(df[column]>upper_bound,upper_bound,df[column])
    outlier_handling_status = "Outliers handled using trimming method."
    return df,outlier_handling_status

#Normalization and Standardization

def normalize_and_standardize(df):
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    
    if not numerical_columns.any():
        return df, "No numerical features found to normalize or standardize."
    
    # Min-Max Scaling
    min_max_scaler = MinMaxScaler()
    df[numerical_columns] = min_max_scaler.fit_transform(df[numerical_columns])
    
    # Z-Score Standardization
    standard_scaler = StandardScaler()
    df[numerical_columns] = standard_scaler.fit_transform(df[numerical_columns])
    
    return df, "Normalization and standardization applied to numerical features."



# Text Cleaning

def text_cleaning(df, text_columns):
    text_cleaned = False  # Flag to indicate if text cleaning was performed
    for column in text_columns:
        # Convert text to lowercase
        df[column] = df[column].str.lower()

        # Remove punctuation
        df[column] = df[column].apply(lambda x: re.sub(r'[' + string.punctuation + ']', '', x))

        # Tokenize text
        df[column] = df[column].apply(word_tokenize)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        df[column] = df[column].apply(lambda tokens: [token for token in tokens if token not in stop_words])

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        df[column] = df[column].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

        # Concatenate tokens back into strings
        df[column] = df[column].apply(lambda tokens: ' '.join(tokens))

        text_cleaned = True

    if text_cleaned:
        cleaning_status = "Text data cleaned."
    else:
        cleaning_status = "No text data found to clean."

    return df, cleaning_status


#Categorical Encoding

def categorical_encoding(df):
    encoded_df = df.copy()
    encoding_status = {}  # Dictionary for encoding status

    # Identify the categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns

    if not categorical_columns.any():
        return encoded_df, "No categorical variables found."

    for column in categorical_columns:
        print(f"Processing column: {column}")  # Debugging print statement
        unique_values_count = df[column].nunique()

        # Encode categorical variables
        if unique_values_count <= 10:
            # One hot encoding for low cardinality
            one_hot_encoder = OneHotEncoder(drop='first')
            encoded_column = one_hot_encoder.fit_transform(encoded_df[[column]])
            encoded_column_df = pd.DataFrame(encoded_column.toarray(),
                                              columns=[f'{column}_{category}' for category in
                                                       one_hot_encoder.categories_[0][1:]],
                                              index=encoded_df.index)  # Preserve index
            encoded_df = pd.concat([encoded_df, encoded_column_df], axis=1)
            encoded_df.drop(column, axis=1, inplace=True)
            encoding_status[column] = "One-hot encoding applied."

        elif 10 < unique_values_count <= 20:
            # Label encoding for moderate cardinality
            label_encoder = LabelEncoder()
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])
            encoding_status[column] = "Label encoding applied."

        else:
            # No encoding for high cardinality
            encoding_status[column] = "No encoding applied due to high cardinality (> 20 unique values)."

    return encoded_df, encoding_status


#Function to handle datetime values

def handle_datetime(df):
    datetime_columns = []  # List to store columns containing datetime values
    datetime_pattern = r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}|\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
    
    # Loop through columns to identify datetime values
    for column in df.columns:
        # Check if any value in the column matches the datetime pattern
        if df[column].apply(lambda x: bool(re.match(datetime_pattern, str(x)))).any():
            datetime_columns.append(column)
    
    if datetime_columns:
        for column in datetime_columns:
            # Attempt to convert the column to datetime format
            try:
                df[column] = pd.to_datetime(df[column])
            except ValueError:
                pass  # Ignore if conversion fails
        
        status = "Datetime values handled."
    else:
        status = "No datetime values found."
    
    return datetime_columns, status






#streamlit interface

def main():
    st.title("Automated Data Cleaner")
    st.write("Upload your dataset here:")
    
    #file uploading logic
    uploaded_file = st.file_uploader("Choose a file",type=['csv','xlsv'])
    
    if uploaded_file is not None:
        # df = pd.read_csv(uploaded_file)
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        #Handling date time values
        datetime_columns, datetime_handling_status = handle_datetime(df)
        st.write(datetime_handling_status)
        
        #Data cleaning 
        #missing val handling
        df, cleaning_status = handle_missing_values(df)
        st.write(cleaning_status)
        
        #Outlier handling
        outlier_handling_status = "No outlier handling performed."  # Default status
        df,outlier_handling_status = handle_outliers(df)
        st.write(outlier_handling_status)
        
        # Normalize and standardize
        df, normalization_status = normalize_and_standardize(df)
        st.write(normalization_status)
                
        # Determine text columns dynamically (you can improve this based on your dataset)
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        
        # Text cleaning
        df, text_cleaning_status = text_cleaning(df, text_columns)
        st.write(text_cleaning_status)
        
        #Categorical encoding
        encoded_df, categorical_encoding_status = categorical_encoding(df)
        st.write(categorical_encoding_status)
        
        st.write(encoded_df.head())  # Display the encoded DataFrame
        
        # Download button
        
        # Prepare file content
        cleaned_data = encoded_df.to_csv(index=False)
            
        # Prompt user to download the file
        st.download_button(
            label="Download",
            data=cleaned_data,
            file_name='cleaned_dataset.csv',
            mime='text/csv'
        )

    
if __name__ == "__main__":
    main()
