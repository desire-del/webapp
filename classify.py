import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# App title
st.title("Classifiers In Action")

# Description
st.text("Choose your Dataset, the classify: test parameters and get your results")

seed = 42

data_ls = ["wine", "iris", "cancer"]
model_ls = ["svm", "knn", "RandomForest"]
def return_data(datasets):
    if datasets == data_ls[0]:
        data = load_wine()
    elif datasets == data_ls[1]:
        data = load_iris()
    else:
        data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names, index= None)
    df["Type"] = data.target
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                         random_state=seed, test_size=0.2)
    return X_train, X_test, y_train, y_test, df, data.target_names


# Side Bar 
sidebar =  st.sidebar
datasets = sidebar.selectbox("Choose a Dataset", options=data_ls)
classifier = sidebar.selectbox("Choose Model", model_ls)
X_train, X_test, y_train, y_test, df, target_names = return_data(datasets.lower().strip())
st.dataframe(df.sample(n=5, random_state=seed))
st.subheader("Classes")

for id, val in enumerate(target_names):
    st.text(f"{id} : {val}")

def getClassifier(classifier):
    if classifier == model_ls[0]:
        c = st.sidebar.slider("Choose a value for C ", min_value=0.0001, max_value=10.0)
        model = SVC(C=c)
    elif classifier == model_ls[1]:
        neighbors = st.sidebar.slider("Number of Neighbors", min_value=1, max_value=10)
        model = KNeighborsClassifier(n_neighbors=neighbors)
    else:
        max_depth = st.sidebar.slider("Max depth", 2, 10)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state=seed)
    return model

def getPCA(df):
    pca = PCA(n_components=3)
    result = pca.fit_transform(df.loc[:,df.columns !="Type"])
    df["pca_1"] = result[:,0]
    df["pca_2"] = result[:,1]
    df["pca_3"] = result[:,2]
    return df

# 2 PCA vizualisation

df = getPCA(df)
fig = plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x = "pca_1",
    y = "pca_2",
    palette=sns.color_palette("hls", len(target_names)),
    hue="Type",
    legend="full"
)
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(projection='3d')
ax2.scatter(
    xs=df["pca_1"],
    ys=df["pca_2"],
    zs=df["pca_3"],
    c=df["Type"]
)
ax2.set_xlabel("PCA One")
ax2.set_ylabel("PCA Two")
ax2.set_zlabel("PCA Three")
plt.title("3-D PCA visualisation")
#plt.xlabel("PCA One")
#plt.ylabel("PCA Two")
#plt.plot()
st.pyplot(fig)
st.pyplot(fig2)

# Training Model
model = getClassifier(classifier)
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
train_score = model.score(X_train, y_train)

# Precision Metrics
metrics_ls = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
metric = sidebar.selectbox("What Metric to plot ?",metrics_ls)

# Plot metric

if metric == metrics_ls[0]:
    disp = ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y= y_test, display_labels=target_names)
    st.pyplot(disp.figure_)
#elif metric == metrics_ls[1]:
    #disp = RocCurveDisplay.from_estimator(estimator=model,X= X_test, y=y_test)
#else:
    #disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)

st.subheader(f"Train score : {round(train_score, 2)}")
st.subheader(f"Test score : {round(test_score, 2)}")
