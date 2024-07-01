import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Titanic-Dataset.csv'
titanic_data = pd.read_csv(file_path)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
titanic_data['Age'] = imputer.fit_transform(titanic_data[['Age']])
titanic_data['Fare'] = imputer.fit_transform(titanic_data[['Fare']])

imputer = SimpleImputer(strategy='most_frequent')
titanic_data['Embarked'] = imputer.fit_transform(titanic_data[['Embarked']]).ravel()

titanic_data.drop(columns=['Cabin'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])

# Select features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = titanic_data[features]
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Train the random forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the models
logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# Function to make predictions
def predict_survival(models, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    input_data = scaler.transform([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    predictions = {name: model.predict(input_data)[0] for name, model in models.items()}
    return predictions

# Streamlit app
st.title('Titanic Survival Prediction')
st.image('titanic.webp', caption='Titanic')
st.write('Enter the details of the passenger to predict if they would have survived the Titanic disaster.')
# Sidebar for input parameters
st.sidebar.header('Input Parameters')
Pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
Sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
Age = st.sidebar.slider('Age', 0, 80, 25)
SibSp = st.sidebar.number_input('Number of Siblings/Spouses Aboard', 0, 10, 0)
Parch = st.sidebar.number_input('Number of Parents/Children Aboard', 0, 10, 0)
Fare = st.sidebar.number_input('Fare Paid', 0.0, 520.0, 32.0)
Embarked = st.sidebar.selectbox('Port of Embarkation', ['Cherbourg', 'Queenstown', 'Southampton'])

# Convert categorical inputs to numerical values
Sex = 1 if Sex == 'Female' else 0
Embarked = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}[Embarked]

if st.sidebar.button('Predict'):
    models = {
        'Logistic Regression': logistic_model,
        'Random Forest': rf_model
    }
    predictions = predict_survival(models, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    logistic_result = 'Survived' if predictions['Logistic Regression'] == 1 else 'Did not survive'
    rf_result = 'Survived' if predictions['Random Forest'] == 1 else 'Did not survive'
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest'],
        'Prediction': [logistic_result, rf_result],
        'Accuracy': [f"{logistic_accuracy:.2f}", f"{rf_accuracy:.2f}"]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.write("## Model Comparison")
    st.table(comparison_df)
    
    # Plot confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(confusion_matrix(y_test, logistic_model.predict(X_test)), annot=True, fmt='d', ax=ax[0])
    ax[0].set_title('Logistic Regression')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    sns.heatmap(confusion_matrix(y_test, rf_model.predict(X_test)), annot=True, fmt='d', ax=ax[1])
    ax[1].set_title('Random Forest')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')
    st.pyplot(fig)
    
    # Plot feature importance for Random Forest
    feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    st.write("Feature Importances (Random Forest):")
    st.bar_chart(feature_importances)
    
    # Plot ROC curves
    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    
    logistic_auc = auc(logistic_fpr, logistic_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(logistic_fpr, logistic_tpr, label=f'Logistic Regression (AUC = {logistic_auc:.2f})')
    ax.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)
    
    # Plot precision-recall curves
    logistic_prec, logistic_rec, _ = precision_recall_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
    rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(logistic_rec, logistic_prec, label='Logistic Regression')
    ax.plot(rf_rec, rf_prec, label='Random Forest')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    st.pyplot(fig)
