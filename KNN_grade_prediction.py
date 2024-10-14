# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:34:36 2024

@author: tosho
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\tosho\Desktop\Data Mining\KNN Model\student_performance.csv')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)

# Drop 'Name' and 'StudentID' from the dataset
if 'Name' in data.columns:
    data.drop('Name', axis=1, inplace=True)
if 'StudentID' in data.columns:
    data.drop('StudentID', axis=1, inplace=True)

# Encode categorical variables
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])

# Prepare features and target variable
X = data.drop('FinalGrade', axis=1)
y = data['FinalGrade']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the k-NN model
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Get user's input
user_id = input("Please enter your Student ID: ")
user_name = input("Please enter your name: ")
attendance_rate = int(input("Enter your attendance out of 7 classes: "))
self_study_hours = int(input("Enter your self-study hours per week: "))
previous_grade = int(input("Enter your previous grade out of 100: "))

# Prepare new data for prediction (excluding StudentID)
new_data = pd.DataFrame({
    'Gender': [0],  # Replace with your actual encoded value for gender
    'AttendanceRate': [attendance_rate * 100 / 7],  # Convert to percentage
    'StudyHoursPerWeek': [self_study_hours],
    'PreviousGrade': [previous_grade],
    'ExtracurricularActivities': [0],  # Replace with your actual encoded value if applicable
    'ParentalSupport': [1]  # Replace with your actual encoded value if applicable
})

# Make predictions
predicted_grade = knn_classifier.predict(new_data)

# Display the result
print(f"{user_name} (Student ID: {user_id}), your predicted final grade is: {predicted_grade[0]}")

# Create the bar chart
grades = [100, predicted_grade[0]]  # Maximum grade and predicted grade
labels = ['Max Final Grade', f"{user_name}'s Predicted Final Grade"]

# Generate the bar chart
plt.bar(labels, grades, color=['blue', 'orange'])
plt.ylim(0, 110)  # Set y-axis limit for better visualization
plt.ylabel('Grade')
plt.title('Comparison of Final Grades')
plt.grid(axis='y')

# Show the plot
plt.show()
