# f1_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

# Title and author
st.title("Formula 1 World Championship Analysis")
st.subheader("Exploring F1 Race Data in 2020-2022")
st.write("**Author:** Chiara Zandomeneghi VR513170")

# Load data
def load_data():
    results = pd.read_csv('results.csv')
    races = pd.read_csv('races.csv')
    constructors = pd.read_csv('constructors.csv')
    drivers = pd.read_csv('drivers.csv')
    return results, races, constructors, drivers

results, races, constructors, drivers = load_data()

# 1 - Explore the dataset
st.header("1 - Explore the Dataset")

# Merge dataframes
races = races[['raceId', 'year', 'name']].rename(columns={'name': 'PrixName'})
merged_df = pd.merge(results, races, on='raceId')

constructors = constructors[['constructorId', 'name']].rename(columns={'name': 'ConstructorName'})
merged_df = pd.merge(merged_df, constructors, on='constructorId')

drivers = drivers[['driverId', 'surname']].rename(columns={'surname': 'DriverSurname'})
merged_df = pd.merge(merged_df, drivers, on='driverId')

# Data Information 
st.write("#### Information")
st.write(merged_df.info())

# Data Preview 
st.write("#### Head")
st.write(merged_df.head())
st.write("#### Tail")
st.write(merged_df.tail())

# Data Description 
st.write("#### Description")
st.write(merged_df.describe())

# Data Shape 
st.write("#### Shape")
st.write(merged_df.shape)

# 2 - Cleaning up the dataset
st.header("2 - Cleaning up the Dataset")

# List of features to analyze
features = ['resultId', 'raceId', 'driverId', 'constructorId', 'number', 'grid',
            'position', 'positionText', 'positionOrder', 'points', 'laps', 'time',
            'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed',
            'statusId']

# Feature Counts 
for feature in features:
    st.write(f"#### {feature} counts:")
    st.write(merged_df[feature].value_counts())

# 3 - Handle Outliers, Missing Values and Drop Columns
st.header("3 - Handle Outliers, Missing Values, and Drop Columns")

# Identify Missing Values 
columns_to_convert = ['resultId', 'raceId', 'driverId', 'constructorId',
                      'number', 'grid', 'position', 'positionText', 'positionOrder',
                      'points', 'laps', 'time', 'milliseconds', 'fastestLap',
                      'rank', 'fastestLapTime', 'fastestLapSpeed', 'statusId']
merged_df[columns_to_convert] = merged_df[columns_to_convert].replace('\\N', np.nan)
missing_values = merged_df.isna().sum()
st.write(missing_values)

# Drop Unnecessary Columns 
df = merged_df.drop(['resultId', 'driverId', 'constructorId', 'number', 'position',
                     'positionText', 'time', 'milliseconds', 'fastestLap',
                     'rank', 'fastestLapTime', 'fastestLapSpeed'], axis=1)
st.write("### Describe")
st.write(df.describe()) 
st.write("### Shape")
st.write(df.shape)

# Remove Columns with Missing Values 
df_clean = df.dropna(axis=1)
df_clean = df[['raceId', 'grid', 'positionOrder', 'points', 'laps', 'statusId',
               'year', 'PrixName', 'DriverSurname', 'ConstructorName']]
st.write("### Data after cleaning and removing outliers")
st.write(df_clean.shape)

# Display Clean Data 
st.write("#### Display Clean Data ")
st.write(df_clean)

# Check for Null Values in Clean Data  
st.write("#### Check for Null Values in Clean Data ")
st.write(df_clean.isnull().sum())










