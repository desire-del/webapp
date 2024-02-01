import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

seed = 42
train_df = pd.read_csv("./data/train.csv")

def process(df):
    # Update sex column to numerical
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    # Fill the nan values in the age column
    df["Age"] = df['Age'].fillna(value = df['Age'].mean())
    # Create a first class column
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    # Create a second class column
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    # Create a second class column
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    # Select the desired features
    df= df[['Sex' , 'Age' , 'FirstClass', 'SecondClass' ,'ThirdClass' , 'Survived']]
    return df 
sel_col = ["Sex","Age","FirstClass","SecondClass","ThirdClass"]
df_process = process(train_df)
features = df_process[sel_col]
target = df_process["Survived"]

#  Train test split
X_train, X_text, y_train, y_test = train_test_split(features, target, train_size=0.7,
                                                    random_state=seed)
# Data Standardization
scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_text)


# Create and train model

model = LogisticRegression()
model.fit(train_features, y_train)
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_pred = model.predict(test_features)

# Calculate the confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

st.title("Would you survived the Titanic Disaster ?")
st.subheader("This model will predict if a passenger would survive the Titanic Disaster or not")

name = st.text_input("The Passenger Name")
sex = st.selectbox("The sex", options=["Male", "Female"])
age = st.slider("Age", 1, 100, 20, 1)
class_list = ['First Class' , 'Second Class' , 'Third Class']
pclass = st.selectbox("Passenger Class",options=class_list)
sex = 0 if sex =="Male" else 1
pclass_encode = [0]*len(class_list)
for i in range(len(class_list)):
    if pclass == class_list[i]:
        pclass_encode[i] = 1
        break
input_data = scaler.transform([[sex, age, *pclass_encode]])
prediction = model.predict(input_data)
prob = model.predict_proba(input_data)
if name != "":
    if prediction[0] == 1:
        st.subheader(f"Great new {name} you have {round(prob[0][1]*100,0)}% of chance to survived")
    else :
        st.subheader(f"Sorry you have {round(prob[0][0]*100,0)}% of chance to not survived")
#st.table(train_df.head(5))
