# Import the Pandas library for creating and manipulating DataFrames
import pandas as pd

# Import the NumPy library for mathematical operations
import numpy as np

# Import preprocessing tools
# LabelEncoder converts text labels into numbers
# OneHotEncoder converts categories into separate binary columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Import feature scaling techniques
# StandardScaler standardizes data
# MinMaxScaler scales data between 0 and 1
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import Machine Learning models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Import evaluation metrics for classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Import confusion matrix and regression evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

# Create a sample dataset using a Python dictionary
data = {

    # Number of hours studied
    "StudyHours":[1,2,3,4,5,6,7,8,9,10],

    # Student attendance percentage
    "Attendance":[50,55,60,65,70,75,80,85,90,95],

    # Whether the assignment was submitted
    "Assignment":["Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes"],

    # Student city
    "City":["Delhi","Delhi","Mumbai","Mumbai","Delhi","Mumbai","Delhi","Mumbai","Delhi","Mumbai"],

    # Target column
    # 0 = Fail
    # 1 = Pass
    "Pass":[0,0,0,1,1,1,1,1,1,1],

    # Marks obtained by the student
    "Marks":[45,55,60,65,70,75,80,85,90,100]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Uncomment this line if you want to display the dataset
# print(df)

# Display the project title
print("\n========== Machine Learning Menu ==========")

# Display menu options
print("1.  LabelEncoder")
print("2.  OneHotEncoder")
print("3.  StandardScaler")
print("4.  MinMaxScaler")
print("5.  Train-Test Split")
print("6.  Linear Regression")
print("7.  Logistic Regression")
print("8.  K-Nearest Neighbors (KNN)")
print("9.  Decision Tree Classifier")
print("10. Accuracy Score")
print("11. Precision Score")
print("12. Recall Score")
print("13. F1 Score")
print("14. Confusion Matrix")
print("15. Mean Absolute Error (MAE)")
print("16. Mean Squared Error (MSE)")
print("17. Root Mean Squared Error (RMSE)")

# Print a closing line for the menu
print("===========================================")

# Ask the user to choose an option
user_choice = int(input("Enter the option : "))

#########################################################
# Option 1 : Label Encoder
#########################################################

if (user_choice == 1):

    # Display the selected option
    print("--- LabelEncoder ---")

    # Select the Assignment column
    X = df[["Assignment"]]

    # Create a LabelEncoder object
    le = LabelEncoder()

    # Convert Yes/No into numbers
    df["Assignment"] = le.fit_transform(X)

    # Display the updated DataFrame
    print(df)

#########################################################
# Option 2 : One-Hot Encoder
#########################################################

elif (user_choice == 2):

    # Display the selected option
    print("--- OneHotEncoded ---")

    # Create a OneHotEncoder object
    le = OneHotEncoder()

    # Convert the City column into binary columns
    new_df = le.fit_transform(df[["City"]]).toarray()

    # Create a DataFrame with meaningful column names
    new_df = pd.DataFrame(
        new_df,
        columns=["City_Delhi", "City_Mumbai"]
    )

    # Remove the original City column
    df = df.drop("City", axis=1)

    # Add the new encoded columns to the original DataFrame
    df = pd.concat([df, new_df], axis=1)

    # Display the updated DataFrame
    print(df)

#########################################################
# Option 3 : Standard Scaler
#########################################################

elif (user_choice == 3):

    # Display the selected option
    print("--- StandardScaler ---")

    # Create a StandardScaler object
    le = StandardScaler()

    # Select the StudyHours column
    X = df[['StudyHours']]

    # Standardize the values
    new_df = le.fit_transform(X)

    # Display the scaled values
    print(new_df)

#########################################################
# Option 4 : Min-Max Scaler
#########################################################

elif (user_choice == 4):

    # Display the selected option
    print("--- MinMaxScaler ---")

    # Create a MinMaxScaler object
    le = MinMaxScaler()

    # Select the StudyHours column
    X = df[['StudyHours']]

    # Scale values between 0 and 1
    new_df = le.fit_transform(X)

    # Display the scaled values
    print(new_df)

#########################################################
# Option 5 : Train-Test Split
#########################################################

elif (user_choice == 5):

    # Display the selected option
    print("--- train_test_split ---")

    # Select the input feature
    X = df[['StudyHours']]

    # Select the target column
    y = df[['Pass']]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Display the training input data
    print(x_train)

    print("\n")

    # Display the testing input data
    print(x_test)

    print("\n")

    # Display the training target data
    print(y_train)

    print("\n")

    # Display the testing target data
    print(y_test)

#########################################################
# Option 6 : Linear Regression
#########################################################

elif (user_choice == 6):

    # Display the selected option
    print("--- LinearRegression ---")

    # Target variable (Marks)
    y = df["Marks"]

    # Input feature (Study Hours)
    X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model using X and y
    model.fit(X, y)

    # Ask the user for study hours
    choice = float(input("Enter the Study Hours : "))

    # Predict the marks for the entered study hours
    pre = model.predict([[choice]])

    # Display the predicted marks
    print("Your Approx Marks :", pre)

#########################################################
# Option 7 : Logistic Regression
#########################################################

elif (user_choice == 7):

    # Display the selected option
    print("--- LogisticRegression ---")

    # Input feature (Study Hours)
    X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

    # Target variable
    # 0 = Fail
    # 1 = Pass
    y = df['Pass']

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X, y)

    # Ask the user for study hours
    choice = int(input("Enter the Study Hours : "))

    # Predict whether the student will pass or fail
    new_pred = model.predict([[choice]])

    # If prediction is 0
    if (new_pred == 0):

        # Display fail message
        print("You likely to be fail")

    # Otherwise
    else:

        # Display pass message
        print("You likely to be Pass")

#########################################################
# Option 8 : K-Nearest Neighbors (KNN)
#########################################################

elif (user_choice == 8):

    # Display the selected option
    print("--- KNeighborsClassifier ---")

    # Input data
    # Column 1 = Study Hours
    # Column 2 = Assignment Marks
    X = [
        [1,33],
        [2,40],
        [3,45],
        [4,50],
        [5,60],
        [6,75],
        [7,80]
    ]

    # Target values
    # 0 = Fail
    # 1 = Pass
    y = [0,0,0,1,1,1,1]

    # Create the KNN model
    model = KNeighborsClassifier()

    # Train the model
    model.fit(X, y)

    # Ask the user for study hours
    choice = int(input("Enter the Study Hours : "))

    # Ask the user for assignment marks
    assignment = int(input("Enter the Assignment marks : "))

    # Predict pass or fail
    new_pred = model.predict([[choice, assignment]])[0]

    # If prediction is Fail
    if (new_pred == 0):

        print("You likely to be fail in Assignment")

    # Otherwise
    else:

        print("You likely to be Pass in Assignment")

#########################################################
# Option 9 : Decision Tree Classifier
#########################################################

elif (user_choice == 9):

    # Display the selected option
    print("--- DecisionTreeClassifier ---")

    # Input data
    # Column 1 = Study Hours
    # Column 2 = Assignment Marks
    X = [
        [1,33],
        [2,40],
        [3,45],
        [4,50],
        [5,60],
        [6,75],
        [7,80]
    ]

    # Target values
    # 0 = Fail
    # 1 = Pass
    y = [0,0,0,1,1,1,1]

    # Create the Decision Tree model
    model = DecisionTreeClassifier()

    # Train the model
    model.fit(X, y)

    # Ask the user for study hours
    choice = int(input("Enter the Study Hours : "))

    # Ask the user for assignment marks
    assignment = int(input("Enter the Assignment marks : "))

    # Predict pass or fail
    new_pred = model.predict([[choice, assignment]])[0]

    # If prediction is Fail
    if (new_pred == 0):

        print("You likely to be fail in Assignment")

    # Otherwise
    else:

        print("You likely to be Pass in Assignment")

#########################################################
# Option 10 : Accuracy Score
#########################################################

elif (user_choice == 10):

    # Display the selected option
    print("--- Accuracy Score ---")

    # Select the input feature
    X = df[["StudyHours"]]

    # Select the target column
    y = df["Pass"]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Predict the output for the test data
    y_pred = model.predict(x_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Display the accuracy
    print("Accuracy Score :", accuracy)

#########################################################
# Option 11 : Precision Score
#########################################################

elif (user_choice == 11):

    # Display the selected option
    print("--- Precision Score ---")

    # Select the input feature
    X = df[["StudyHours"]]

    # Select the target column
    y = df["Pass"]

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Predict the output
    y_pred = model.predict(x_test)

    # Calculate the precision score
    precision = precision_score(y_test, y_pred)

    # Display the precision score
    print("Precision Score:", precision)

#########################################################
# Option 12 : Recall Score
#########################################################

elif (user_choice == 12):

    # Display the selected option
    print("--- Recall Score ---")

    # Select the input feature
    X = df[["StudyHours"]]

    # Select the target column
    y = df["Pass"]

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Predict the output
    y_pred = model.predict(x_test)

    # Calculate the recall score
    Recall = recall_score(y_test, y_pred)

    # Display the recall score
    print("Recall Score:", Recall)

#########################################################
# Option 13 : F1 Score
#########################################################

elif (user_choice == 13):

    # Display the selected option
    print("--- F1 Score ---")

    # Select the input feature
    X = df[["StudyHours"]]

    # Select the target column
    y = df["Pass"]

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(x_train, y_train)

    # Predict the output
    y_pred = model.predict(x_test)

    # Calculate the F1 score
    F1 = f1_score(y_test, y_pred)

    # Display the F1 score
    print("F1 Score:", F1)

#########################################################
# Option 14 : Confusion Matrix
#########################################################

elif (user_choice == 14):

    # Display the selected option
    print("--- Confusion matrix ---")

    # Actual values
    score = df["Pass"]

    # Predicted values
    pre_score = [0,0,1,1,0,1,0,1,1,1]

    # Create the confusion matrix
    cm = confusion_matrix(
        y_true=score,
        y_pred=pre_score
    )

    # Display the confusion matrix
    print(cm)

#########################################################
# Option 15 : Mean Absolute Error (MAE)
#########################################################

elif (user_choice == 15):

    # Display the selected option
    print("Mean Absolute Error")

    # Actual values
    score = df["Pass"]

    # Predicted values
    pre_score = [0,0,1,1,0,1,0,1,1,1]

    # Calculate the Mean Absolute Error
    mae = mean_absolute_error(score, pre_score)

    # Display the MAE
    print(mae)

#########################################################
# Option 16 : Mean Squared Error (MSE)
#########################################################

elif (user_choice == 16):

    # Display the selected option
    print("Mean Square Error")

    # Actual values
    score = df["Pass"]

    # Predicted values
    pre_score = [0,0,1,1,0,1,0,1,1,1]

    # Calculate the Mean Squared Error
    mse = mean_squared_error(score, pre_score)

    # Display the MSE
    print(mse)

#########################################################
# Option 17 : Root Mean Squared Error (RMSE)
#########################################################

elif (user_choice == 17):

    # Display the selected option
    print("Root Mean Suare Error")

    # Actual values
    score = df["Pass"]

    # Predicted values
    pre_score = [0,0,1,1,0,1,0,1,1,1]

    # Calculate the Root Mean Squared Error
    rmse = root_mean_squared_error(score, pre_score)

    # Display the RMSE
    print(rmse)

#########################################################
# Invalid Menu Option
#########################################################

else:

    # Display an error message if the user enters an invalid option
    print("Invalid option : Try Again")