import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#Read the csv file
data = pd.read_csv("CropPredict/dataset/Crop_recommendation.csv")
print(data.head())

# data.isnull().sum()

#split the data into features and target variable
x = data.drop(columns=['label'])
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

#Train the model
model = RandomForestClassifier()
model.fit(x_train,y_train)
print("Model trained successfully")

# Save the model as a pickle file
pickle.dump(model, open("CropPredict/cropmodel.pkl", "wb"))
print("pickle file created successfully")

#Predict on test data
y_pred= model.predict(x_test)

#Evaluate the model
acc = model.score(x_test,y_test)
print(acc)

#Predict crop for given input data
new_features = [[117 ,32,34,26.2724184,52.12739421,6.758792552,127.1752928,]]
predicted_crop = model.predict(new_features) #x_test
print(predicted_crop) #y_test