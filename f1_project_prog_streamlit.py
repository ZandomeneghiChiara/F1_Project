# f1_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np

# Title and author
st.title("Formula 1 World Championship Analysis")
st.subheader("Exploring F1 Race Data in 2020-2022")
st.write("**Author:** Chiara Zandomeneghi VR513170")
st.markdown("""
    1. Identification Information
    - resultId: Unique identifier for the race result entry.
    - raceId: Unique identifier for the race.
    - driverId: Unique identifier for the driver.
    - constructorId: Unique identifier for the constructor/team.
    2. Driver and Constructor Information
    - number: The race number of the driver for that event.
    3. Starting and Finishing Positions
    - grid: Starting position of the driver on the grid.
    - position: Finishing position of the driver in the race.
    - positionText: Textual representation of the finishing position (e.g., "1", "DNF").
    - positionOrder: Numerical order of finishing positions.
    4. Performance Metrics
    - points: The points awarded to the driver for this race.
    - laps: Number of laps completed by the driver.
    - time: Total time taken to complete the race.
    - milliseconds: Total race time in milliseconds.
    - fastestLap: Lap number on which the driver set their fastest lap.
    - rank: Rank of the fastest lap within the race.
    - fastestLapTime: Time of the driver’s fastest lap.
    - fastestLapSpeed: Average speed during the driver’s fastest lap.
    5. Race Status
    - statusId: Unique identifier indicating the race status (e.g., finished, retired, disqualified).""")


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

# 4 - Plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

st.header("4 - Plots")
# Filter data for the years 2020 to 2022
df_filtered = df[(df['year'] >= 2020) & (df['year'] <= 2022)]
st.write("#### Filter data for the years 2020 to 2022 ")
st.write(df_filtered)

# Display counts for various features
st.write("\nraceId_f counts:")
st.write(df_filtered['raceId'].value_counts().shape)  # 61

st.write("\ngrid_f counts:")
st.write(df_filtered['grid'].value_counts().shape)  # 21

st.write("\npositionOrder_f counts:")
st.write(df_filtered['positionOrder'].value_counts().shape)  # 20

st.write("\nPrixName_f counts:")
st.write(df_filtered['PrixName'].value_counts().shape)  # 32

st.write("\nConstructorName_f counts:")
st.write(df_filtered['ConstructorName'].value_counts())  # 12

st.write("\nDriverSurname_f counts:")
st.write(df_filtered['DriverSurname'].value_counts().shape)  # 30

# 4.1 - Podium Finishes by Constructor
st.subheader("4.1 - Podium Finishes by Constructor")

# 1. Race Wins for each Constructors
constructors = df_filtered['ConstructorName'].unique()
# Loop through each constructor and create a count plot
for constructor in constructors:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df_filtered[df_filtered['ConstructorName'] == constructor],
                  x='positionOrder', palette='viridis', ax=ax)
    ax.set_title(f'Race Wins by {constructor}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Count')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    plt.close(fig)

# 2. Race Wins by Constructors
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df_filtered, x='ConstructorName', hue='positionOrder',
              order=df_filtered['ConstructorName'].value_counts().index[:12], ax=ax)
ax.set_title('Race Wins by Constructors')
ax.set_xlabel('Constructor')
ax.set_ylabel('Count')
ax.legend(title='Position')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)
plt.close(fig)


# 4.2 - Podium Finishes by Drivers
st.subheader("4.2 - Podium Finishes by Drivers")

# 3. Race Wins for each Driver
drivers = df_filtered['DriverSurname'].unique()
# Loop through each driver and create a count plot
for driver in drivers:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df_filtered[df_filtered['DriverSurname'] == driver],
                  x='positionOrder', palette='viridis', ax=ax)
    ax.set_title(f'Race Wins by {driver}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Count')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    plt.close(fig)

# 4. Podium Finishes by Top Drivers
top_drivers = df_filtered['DriverSurname'].value_counts().index[:30]
fig, ax = plt.subplots(figsize=(15, 6))
sns.countplot(data=df_filtered[df_filtered['DriverSurname'].isin(top_drivers)], x='DriverSurname', hue='positionOrder', ax=ax)
ax.set_title('Podium Finishes by Top Drivers')
ax.set_xlabel('Driver')
ax.set_ylabel('Count')
ax.legend(title='Position')
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
st.pyplot(fig)
plt.close(fig)


# 4.3 - Points Scored by Each Constructor Over Time
st.subheader("4.3 - Points Scored by Each Constructor Over Time")
# 5. Points Scored by Each Constructor Over Time
# Get the unique constructor names from df_filtered
constructors = df_filtered['ConstructorName'].unique()
# Iterate over each constructor to create and display a line plot
for constructor in constructors:
    constructors_df = df_filtered[df_filtered['ConstructorName'] == constructor]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=constructors_df, x='raceId', y='points', marker='o', ax=ax)
    ax.set_title(f'Points Scored by {constructor} Over Time')
    ax.set_xlabel('Race ID')
    ax.set_ylabel('Points')
    ax.grid(True)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    plt.close(fig)


# 4.4 - Points Scored by Each Driver Over Time
st.subheader("4.4 - Points Scored by Each Driver Over Time")
# 6. Points Scored by Each Driver Over Time 
# Get the unique driver surnames
drivers = df_filtered['DriverSurname'].unique()
# Iterate over each driver to create and display a line plot
for driver in drivers:
    driver_df = df_filtered[df_filtered['DriverSurname'] == driver]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=driver_df, x='raceId', y='points', marker='o', ax=ax)
    ax.set_title(f'Points Scored by {driver} Over Time')
    ax.set_xlabel('Race ID')
    ax.set_ylabel('Points')
    ax.grid(True)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    plt.close(fig) 


# 4.5 - Points Distribution by Prix Name
st.subheader("4.5 - Points Distribution by Prix Name")
# 7. Points Distribution by Prix Name
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=df, x='PrixName', y='points', ax=ax)
ax.set_title('Points Distribution by Grand Prix', fontsize=16)
ax.set_xlabel('Grand Prix Name', fontsize=14)
ax.set_ylabel('Points', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)
plt.close(fig)


# 4.6 - Average Points by Constructor and Year
st.subheader("4.6 - Average Points by Constructor and Year")
# 8. Average Points by Constructor and Year
# Set the size of the figure
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_filtered, x='year', y='points', hue='ConstructorName', estimator='mean', ax=ax)
ax.set_title('Average Points by Constructor and Year', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Average Points', fontsize=14)
ax.legend(title='Constructor', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)
plt.close(fig)


# 4.7 - Average Points by Driver and Year
st.subheader("4.7 - Average Points by Driver and Year")
# 9. Average Points by Driver and Year
plt.figure(figsize=(12, 7))
sns.barplot(data=df_filtered, x='year', y='points', hue='DriverSurname', estimator='mean')
plt.title('Average Points by Driver and Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Points', fontsize=14)
plt.legend(title='Driver', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)


# 4.8 - Constructor Performance Comparison by Year
st.subheader("4.8 - Constructor Performance Comparison by Year")
# 10. Constructor Performance Comparison by Year
# Identify top 5 constructors based on frequency
top_constructors = df_filtered['ConstructorName'].value_counts().index[:5]
# Filter data for only the top constructors
top_constructors_data = df_filtered[df_filtered['ConstructorName'].isin(top_constructors)]
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=top_constructors_data, x='year', y='points', hue='ConstructorName', ax=ax)
ax.set_title('Constructor Performance Comparison by Year', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Points', fontsize=14)
ax.legend(title='Constructor', loc='upper left')
ax.grid(True)
st.pyplot(fig)
plt.close(fig)


# 4.9 - Driver Performance Comparison by Year
st.subheader("4.9 - Driver Performance Comparison by Year")
# 11. Driver Performance Comparison by Year
# Identify top 5 drivers based on frequency
top_drivers = df_filtered['DriverSurname'].value_counts().index[:5]
# Filter data for only the top drivers
top_drivers_data = df_filtered[df_filtered['DriverSurname'].isin(top_drivers)]
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=top_drivers_data, x='year', y='points', hue='DriverSurname', ax=ax)
ax.set_title('Driver Performance Comparison by Year', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Points', fontsize=14)
ax.legend(title='Driver', loc='upper left') 
ax.grid(True) 
st.pyplot(fig)
plt.close(fig)













