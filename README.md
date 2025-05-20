# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MAHIR HUSSAIN S
RegisterNumber:  212223040109
*/
```
```
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Step 1: Detect encoding
file = r"C:\Users\admin\Downloads\spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print("Detected Encoding:", result)

# Step 2: Load CSV with correct encoding
data = pd.read_csv(file, encoding='windows-1252')

# Step 3: Quick data check
print(data.head())
print(data.info())
print(data.isnull().sum())

# Step 4: Use correct columns (v1 = label, v2 = message)
x = data["v2"].values  # Message text
y = data["v1"].values  # Labels (ham/spam)

# Step 5: Encode labels
le = LabelEncoder()
y = le.fit_transform(y)  # ham = 0, spam = 1

# Step 6: Split into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 7: Convert text to feature vectors
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Step 8: Train the SVM model
svc = SVC()
svc.fit(x_train, y_train)

# Step 9: Predictions and evaluation
y_pred = svc.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Optional: Detailed metrics
print(metrics.classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
```
## Output:
![image](https://github.com/user-attachments/assets/cf94b2a5-ddd6-4406-92c9-686f6ec2ea91)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
