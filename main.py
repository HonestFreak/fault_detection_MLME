from fastapi import FastAPI, HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as Xgb
from sklearn.svm import SVC
import io
import base64
from pydantic import BaseModel
import numpy as np


app = FastAPI(docs_url="/")

# Load the dataset
df_class = pd.read_csv("sample_data/classData.csv")
# df_class = pd.read_csv("sample_data/detect_dataset.csv")


# Change path to the dataset
@app.get("/data/change_path")
def change_path(path: str):
    global df_class
    df_class = pd.read_csv(path)
    return {"message": "Path changed successfully"}

# Get information about the data
@app.get("/data/info")
def data_info():
    data_info = {
        "columns": list(df_class.columns),
        "total_entries": len(df_class),
        "data_types": df_class.dtypes.apply(lambda x: str(x)).to_dict(),
        "non_null_count": df_class.count().to_dict()
    }
    return data_info


# Get total null content
@app.get("/data/null")
def total_null_parameters_in_the_data():
    return int(df_class.isnull().sum().sum())

# Get shape of the data
@app.get("/data/shape")
def data_shape():
    return df_class.shape

# Countplot and Pie Chart for a given column
@app.get("/visualization/{column}")
def visualization(column: str):
    if column not in df_class.columns:
        raise HTTPException(status_code=404, detail="Column not found")
    else:
        fig, axs = plt.subplots(1, 2)
        sns.countplot(x=column, data=df_class, ax=axs[0])
        df_class[column].value_counts().plot.pie(explode=[0.1] * df_class[column].nunique(), autopct='%1.2f%%', shadow=True, ax=axs[1])
        plt.title(column + " Visualization", fontsize=20, color='Brown', pad=20)
        
        # Save plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Encode plot to base64 string
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {"plot": plot_base64}

# Encode fault types
@app.get("/data/encode_fault_types")
def encode_fault_types():
    df_class['Fault_Type'] = df_class['G'].astype('str') + df_class['C'].astype('str') + df_class['B'].astype('str') + df_class['A'].astype('str')
    df_class['Fault_Type'][df_class['Fault_Type'] == '0000'] = 'NO Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '1001'] = 'Line A to Ground Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '0110'] = 'Line B to Line C Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '1011'] = 'Line A Line B to Ground Fault'
    df_class['Fault_Type'][df_class['Fault_Type'] == '0111'] = 'Line A Line B Line C'
    df_class['Fault_Type'][df_class['Fault_Type'] == '1111'] = 'Line A Line B Line C to Ground Fault'
    return df_class['Fault_Type']

# Train models and evaluate
random_forest = RandomForestClassifier(n_estimators=100)
logreg = LogisticRegression()
decision = DecisionTreeClassifier()    
xgb = Xgb.XGBClassifier()
svc = SVC()

@app.get("/models")
def train_models():
    encoder = LabelEncoder()
    df_class['Fault_Type'] = encoder.fit_transform(df_class['Fault_Type'])
    X = df_class[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]  # Selecting only sensor readings as features
    y = df_class['Fault_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=21)

    models = []

    # Logistic Regression
    logreg.fit(X_train, y_train)
    log_accuracy = round(logreg.score(X_test, y_test) * 100, 2)
    models.append({"Model": "Logistic Regression", "Model Accuracy Score": log_accuracy})

    # Decision Tree
    decision.fit(X_train, y_train)
    decision_accuracy = round(decision.score(X_test, y_test) * 100, 2)
    models.append({"Model": "Decision Tree", "Model Accuracy Score": decision_accuracy})

    # Random Forest
    random_forest.fit(X_train, y_train)
    random_forest_accuracy = round(random_forest.score(X_test, y_test) * 100, 2)
    models.append({"Model": "Random Forest", "Model Accuracy Score": random_forest_accuracy})

    # XGBoost
    xgb.fit(X_train, y_train)
    xgb_accuracy = round(xgb.score(X_test, y_test) * 100, 2)
    models.append({"Model": "XGBClassifier", "Model Accuracy Score": xgb_accuracy})

    # Support Vector Machines
    svc.fit(X_train, y_train)
    svc_accuracy = round(svc.score(X_test, y_test) * 100, 2)
    models.append({"Model": "Support Vector Machines", "Model Accuracy Score": svc_accuracy})

    return models


class PredictionInput(BaseModel):
    Ia: float
    Ib: float
    Ic: float
    Va: float
    Vb: float
    Vc: float

@app.post("/predict")
def predict_fault_type(input_data: PredictionInput , model: str = "decision"):
    # Prepare input data for prediction
    input_features = np.array([[
        input_data.Ia, input_data.Ib, input_data.Ic,
        input_data.Va, input_data.Vb, input_data.Vc,
    ]])

    # Make prediction using the trained model
    if model == "random_forest":
        predicted_fault_type = random_forest.predict(input_features)
    elif model == "logreg":
        predicted_fault_type = logreg.predict(input_features)
    elif model == "xgb":
        predicted_fault_type = xgb.predict(input_features)
    elif model == "svc":
        predicted_fault_type = svc.predict(input_features)
    else:
        predicted_fault_type = decision.predict(input_features)

    # Map numerical labels back to fault types
    fault_type_mapping = {
        0: 'NO Fault',
        1: 'Line A to Ground Fault',
        2: 'Line B to Line C Fault',
        3: 'Line A Line B to Ground Fault',
        4: 'Line A Line B Line C',
        5: 'Line A Line B Line C to Ground Fault'
    }

    predicted_fault = fault_type_mapping[predicted_fault_type[0]]

    return {"predicted_fault_type": predicted_fault}