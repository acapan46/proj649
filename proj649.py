#!/usr/bin/env python
# coding: utf-8
pip install -r requirements.txt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
from sklearn.preprocessing import StandardScaler
from PIL import Image

st.title("Group Project CSC649")
st.title("Salary Prediction")
st.write(" ")
st.write(" ")
st.subheader("UiTM TAPAH")
st.subheader("A4CS2305A")
st.write(" ")
st.write(" ")
st.write("MUHAMMAD IKMAL BIN ISMAIL           (2022912399)")
st.write("MUHAMMAD ASHRAF BIN AZAHARI         (2022995637)")
st.write("MUHAMMAD ARIF ZIKRI BIN MOHD AFIZI  (2022978157)")
st.write("NURUL NAJIHAH BINTI DZULKIFLI       (2022949525)")
st.write(" ")
st.write(" ")
st.subheader("This system predicts salary classes or ranges using various inputs such as age, location, job title. The algorithms used in this system is Support Vector Machine (SVM), K-Nearest Neighbor(KNN) and Random Forest. The datasets used for the system are from general working adults, web developers, data science industries and white house staff. The performance of the model is measured using the accuracy score.")
image = Image.open('Home.jpg')
st.image(image, caption='Essential Workers')
results_dict = {}

def evaluate_model(predictions, model_name):
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"{model_name} Accuracy Score: {accuracy:.2f}")
    results_dict[model_name] = {'Accuracy': accuracy}

pickData = st.sidebar.selectbox("Choose Dataset", options = ["Home","General Adults","Web Developers","Data Science","White House Staff"])

if pickData == "General Adults":
    
    image = Image.open('adults.jpg')
    st.image(image, caption='General Working Adults')
    
    st.subheader("General Adults Income (Full)")
    data = pd.read_csv("salary.csv",na_values=[' ?'])
    
    data
    desired_size = 500
    data = data.head(desired_size)
    
    st.subheader("Data Preprocessing and Cleaning")
    
    
    st.write("Data NaN value check:")
    st.dataframe(pd.DataFrame({'h': data.nunique().index , 'null':  data.isnull().sum()}))
    
    
    st.write("NaN values will be removed...")
    data['workclass'].fillna(data['workclass'].mode()[0] , axis=0 ,inplace=True)
    data['occupation'].fillna(data['occupation'].mode()[0] , axis=0 ,inplace=True)
    data['native-country'].fillna(data['native-country'].mode()[0] , axis=0 ,inplace=True)
    
    
    st.write("Check Any Duplicates")
    st.write("Duplicates: ",data.duplicated().sum())

    st.write("After NaN values removal")
    st.dataframe(pd.DataFrame({'h': data.nunique().index , 'null':  data.isnull().sum()}))
    
    
    st.write("Encoding for categorical variable")
    le = LabelEncoder()
    categorical_cols = ['workclass','sex', 'marital-status', 'occupation', 'education','native-country']
    data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))

    st.write("Removing categorical variable from dataset.....")
    data = data.drop(['relationship','race'], axis=1)
    
    st.write("Final Dataset:")
    st.dataframe(data)
    
    
    st.subheader("Data input")
    X = data.drop('salary', axis=1)
    X
    
    st.subheader("Data target")
    y = data['salary']
    y
    
    st.subheader("Splitting data with train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    
    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train
    
    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    X_test
    st.write("Training Data Target")
    y_test
    
    model = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])
    
    if model == "Support Vector Machine":
        
        st.subheader("Support Vector Machine Prediction Model")
        st.write(" ")
        
        svm_linear_model = SVC(kernel='linear')
        st.write("Training process...")
        svm_linear_model.fit(X_train, y_train)
        
        st.write("Training Success...")
            
        svm_linear_pred = svm_linear_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_linear_pred
        
        
        st.write(evaluate_model(svm_linear_pred, "SVM Linear"))
        
        
        st.write(" ")
        st.subheader("Poly Kernel")
        svm_poly_model = SVC(kernel='poly', degree=2)
        st.write("Training process...")
        svm_poly_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Poly Prediction with Testing dataset")
        svm_poly_pred = svm_poly_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_poly_pred
        
        
        st.write(evaluate_model(svm_poly_pred, "SVM Poly"))
                 
        st.write(" ")
        st.subheader("RBF Kernel")
        svm_rbf_model = SVC(kernel='rbf')
        st.write("Training process...")
        svm_rbf_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM RBF Prediction with Testing dataset")
        svm_rbf_pred = svm_rbf_model.predict(X_test)
        st.write("Predictions from RBF Kernel Model for Testing Dataset: ")
        svm_rbf_pred
        
        
        st.write(evaluate_model(svm_rbf_pred, "SVM RBF"))
        
        
        st.write(" ")
        st.subheader("Sigmoid Kernel")
        svm_sigmoid_model = SVC(kernel='sigmoid')
        st.write("Training process...")
        svm_sigmoid_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Sigmoid Prediction with Testing dataset")
        svm_sigmoid_pred = svm_sigmoid_model.predict(X_test)
        st.write("Predictions from Sigmoid Kernel Model for Testing Dataset: ")
        svm_sigmoid_pred
        
        
        st.write(evaluate_model(svm_sigmoid_pred, "SVM Sigmoid"))
        
    elif model == "K-Nearest Neighbors":    
        neighbors = [10,50,100,200]
        
        for n in neighbors:
            st.subheader(f"K-Nearest Neighbors model (n_neighbors = {n})")
            knn_model = KNeighborsClassifier(n_neighbors= n)
            st.write("Training the Model...")
            knn_model.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            knn_pred = knn_model.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            knn_pred
            
            st.write(evaluate_model(knn_pred, f"KNN model (n = {n})"))
    
    elif model == "Random Forest":
        
        st.subheader("Random Forest Salary Prediction model")
        
        n = [50, 100, 150, 200]
        for n in n:
            st.write("N_Estimator =",n)            

            rf_model = RandomForestClassifier(n_estimators= n, random_state=42)
            st.write("Training Process...")
            rf_model.fit(X_train, y_train)

            st.write("Training Success...")
            rf_pred = rf_model.predict(X_test)
            st.write("Predictions for Testing Dataset: ")
            rf_pred
            
            st.write(evaluate_model(rf_pred, f"Random Forest (n = {n})"))
        
    
    con = st.sidebar.selectbox("Select Summary", options = ["Select Summary", "Summary"])
    
    if con == "Summary":
        st.subheader("Conclusions from all Model: ")
        results_df = pd.DataFrame(results_dict).transpose()
        results_df.index.name = 'Model'
        results_df.reset_index(inplace=True)

        # Display the results table
        st.table(results_df)
        
        
        
        
        
        
        
        
        
        
elif pickData == "Web Developers":
    
    image = Image.open('dev.jpg')
    
    st.image(image, caption='Web developers')
    
    st.subheader("Web Developers (Full)")
    data = pd.read_csv("web_dev.csv")
    
    data
    
    st.subheader("Data input")
    X = data[['id', 'easy_apply', 'company_rating', 'min_salary', 'max_salary']]
    X
    
    st.subheader("Data target")
    y = data['salary']
    y
    
    st.write("Convert the salary values to numerical format (remove any non-numeric characters)")
    y = y.str.replace('[^\d.]', '', regex=True).astype(float)
    
    st.subheader("Define your criteria for categorizing salaries into classes")
    
    def categorize_salary(salary):
        if salary < 50000:
            return 'low'
        elif salary >= 50000 and salary < 80000:
            return 'medium'
        else:
            return 'high'
        
    y = y.apply(categorize_salary)
    y
    
    from sklearn.impute import SimpleImputer
    st.write("Handle missing values in X")
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    st.subheader("Splitting data with train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train
    
    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    X_test
    st.write("Training Data Target")
    y_test
    
    st.write("Scale the numerical attributes for better performance")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train
    X_test = scaler.transform(X_test)
    X_test
    
    
    model = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])
    
    if model == "Support Vector Machine":
        
        st.subheader("Support Vector Machine Prediction Model")
        st.write(" ")
        
        svm_linear_model = SVC(kernel='linear')
        st.write("Training process...")
        svm_linear_model.fit(X_train, y_train)
        
        st.write("Training Success...")
            
        svm_linear_pred = svm_linear_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_linear_pred
        
        
        st.write(evaluate_model(svm_linear_pred, "SVM Linear"))
        
        
        st.write(" ")
        st.subheader("Poly Kernel")
        svm_poly_model = SVC(kernel='poly', degree=2)
        st.write("Training process...")
        svm_poly_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Poly Prediction with Testing dataset")
        svm_poly_pred = svm_poly_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_poly_pred
        
        
        st.write(evaluate_model(svm_poly_pred, "SVM Poly"))
                 
        st.write(" ")
        st.subheader("RBF Kernel")
        svm_rbf_model = SVC(kernel='rbf')
        st.write("Training process...")
        svm_rbf_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM RBF Prediction with Testing dataset")
        svm_rbf_pred = svm_rbf_model.predict(X_test)
        st.write("Predictions from RBF Kernel Model for Testing Dataset: ")
        svm_rbf_pred
        
        
        st.write(evaluate_model(svm_rbf_pred, "SVM RBF"))
        
        
        st.write(" ")
        st.subheader("Sigmoid Kernel")
        svm_sigmoid_model = SVC(kernel='sigmoid')
        st.write("Training process...")
        svm_sigmoid_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Sigmoid Prediction with Testing dataset")
        svm_sigmoid_pred = svm_sigmoid_model.predict(X_test)
        st.write("Predictions from Sigmoid Kernel Model for Testing Dataset: ")
        svm_sigmoid_pred
        
        
        st.write(evaluate_model(svm_sigmoid_pred, "SVM Sigmoid"))
        
    elif model == "K-Nearest Neighbors":
        
        neighbors = [50,100,200]
        
        for n in neighbors:
            st.subheader(f"K-Nearest Neighbors model (n_neighbors = {n})")
            knn_model = KNeighborsClassifier(n_neighbors= n)
            st.write("Training the Model...")
            knn_model.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            knn_pred = knn_model.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            knn_pred
            
            st.write(evaluate_model(knn_pred, f"KNN model (n = {n})"))
    
    elif model == "Random Forest":
        
        st.subheader("Random Forest Salary Prediction model")
        
        n = [50, 100, 200]
        for n in n:
            st.write("N_Estimator =",n)            

            rf_model = RandomForestClassifier(n_estimators= n, random_state=42)
            st.write("Training Process...")
            rf_model.fit(X_train, y_train)

            st.write("Training Success...")
            rf_pred = rf_model.predict(X_test)
            st.write("Predictions for Testing Dataset: ")
            rf_pred
            
            st.write(evaluate_model(rf_pred, f"Random Forest (n = {n})"))
            
    
    con = st.sidebar.selectbox("Select Summary", options = ["Select Summary", "Summary"])
    if con == "Summary":
        st.subheader("Conclusions from all Model: ")
        results_df = pd.DataFrame(results_dict).transpose()
        results_df.index.name = 'Model'
        results_df.reset_index(inplace=True)

        # Display the results table
        st.table(results_df)
        
        

        
        
        
        
        
        
        
        
        
elif pickData == "Data Science":
    
    image = Image.open('data.jpg')
    
    st.image(image, caption='Data Science Industry')
    
    st.subheader("Data Science")
    
    data = pd.read_csv("salary_data_cleaned.csv")
    data
    
    st.write("Encoding for categorical variable")
    le = LabelEncoder()
    categorical_cols = ['Job Title','Rating','Location','Type of ownership','Industry','Sector']
    data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
    
    st.subheader("Data input")
    X = data.drop(columns = ['Salary Estimate','Job Description','Company Name','Headquarters','Size','Founded','Revenue','Competitors','hourly','employer_provided','company_txt','job_state','same_state','age'])
    X
    
    st.subheader("Data target")
    y = data['Salary Estimate']
    y
    
    st.subheader("Splitting data with train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    
    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train
    
    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    X_test
    st.write("Training Data Target")
    y_test
    
    model = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])
    
    if model == "Support Vector Machine":
        
        st.subheader("Support Vector Machine Prediction Model")
        st.write(" ")
        
        svm_linear_model = SVC(kernel='linear')
        st.write("Training process...")
        svm_linear_model.fit(X_train, y_train)
        
        st.write("Training Success...")
            
        svm_linear_pred = svm_linear_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_linear_pred
        
        
        st.write(evaluate_model(svm_linear_pred, "SVM Linear"))
        
        
        st.write(" ")
        st.subheader("Poly Kernel")
        svm_poly_model = SVC(kernel='poly', degree=2)
        st.write("Training process...")
        svm_poly_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Poly Prediction with Testing dataset")
        svm_poly_pred = svm_poly_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_poly_pred
        
        
        st.write(evaluate_model(svm_poly_pred, "SVM Poly"))
                 
        st.write(" ")
        st.subheader("RBF Kernel")
        svm_rbf_model = SVC(kernel='rbf')
        st.write("Training process...")
        svm_rbf_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM RBF Prediction with Testing dataset")
        svm_rbf_pred = svm_rbf_model.predict(X_test)
        st.write("Predictions from RBF Kernel Model for Testing Dataset: ")
        svm_rbf_pred
        
        
        st.write(evaluate_model(svm_rbf_pred, "SVM RBF"))
        
        
        st.write(" ")
        st.subheader("Sigmoid Kernel")
        svm_sigmoid_model = SVC(kernel='sigmoid')
        st.write("Training process...")
        svm_sigmoid_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Sigmoid Prediction with Testing dataset")
        svm_sigmoid_pred = svm_sigmoid_model.predict(X_test)
        st.write("Predictions from Sigmoid Kernel Model for Testing Dataset: ")
        svm_sigmoid_pred
        
        
        st.write(evaluate_model(svm_sigmoid_pred, "SVM Sigmoid"))
        
    elif model == "K-Nearest Neighbors":
        
        neighbors = [27,51,99]
        
        for n in neighbors:
            st.subheader(f"K-Nearest Neighbors model (n_neighbors = {n})")
            knn_model = KNeighborsClassifier(n_neighbors= n)
            st.write("Training the Model...")
            knn_model.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            knn_pred = knn_model.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            knn_pred
            
            st.write(evaluate_model(knn_pred, f"KNN model (n = {n})"))
    
    elif model == "Random Forest":
        
        st.subheader("Random Forest Salary Prediction model")
        
        n = [10, 50, 100]
        for n in n:
            st.write("N_Estimator =",n)            

            rf_model = RandomForestClassifier(n_estimators= n, random_state=42)
            st.write("Training Process...")
            rf_model.fit(X_train, y_train)

            st.write("Training Success...")
            rf_pred = rf_model.predict(X_test)
            st.write("Predictions for Testing Dataset: ")
            rf_pred
            
            st.write(evaluate_model(rf_pred, f"Random Forest (n = {n})"))
            
    con = st.sidebar.selectbox("Select Summary", options = ["Select Summary", "Summary"])
    if con == "Summary":
        st.subheader("Conclusions from all Model: ")
        results_df = pd.DataFrame(results_dict).transpose()
        results_df.index.name = 'Model'
        results_df.reset_index(inplace=True)

        # Display the results table
        st.table(results_df)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
elif pickData == ("White House Staff"):
    
    image = Image.open('staff.jpg')
    
    st.image(image, caption='Staff workers')
    
    st.subheader("White House Staff Income")
    data = pd.read_csv('wh_staff_dataset.csv')
    
    data
    
    desired_size = 500
    data = data.head(desired_size)
    
    st.subheader("Data Preprocessing and Cleaning")
    
    
    st.write("Fill missing salary values with the mean:")
    mean_salary = data['salary'].mean()
    data['salary'].fillna(mean_salary, inplace=True)
    
    
    st.write("Define salary ranges and convert to class labels")
    salary_ranges = [0, 30000, 50000, 70000, 100000, float('inf')]
    class_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    data['salary_class'] = pd.cut(data['salary'], bins=salary_ranges, labels=class_labels, right=False)
    
    st.write("Final Dataset:")
    st.dataframe(data)
    
    
    st.subheader("Data input")
    X = data[['year', 'gender', 'status', 'pay_basis', 'position_title']]
    X
    
    st.subheader("Data target")
    y = data['salary_class']
    y
    
    st.write("Convert categorical variables to numerical using one-hot encoding...")
    X = pd.get_dummies(X)
    X
    
    st.subheader("Splitting data with train_test_split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    X_train
    st.write("Training Data Target")
    y_train
    
    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    X_test
    st.write("Training Data Target")
    y_test
    
    model = st.sidebar.selectbox ("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest"])
    
    if model == "Support Vector Machine":
        
        st.subheader("Support Vector Machine Prediction Model")
        st.write(" ")
        
        svm_linear_model = SVC(kernel='linear')
        st.write("Training process...")
        svm_linear_model.fit(X_train, y_train)
        
        st.write("Training Success...")
            
        svm_linear_pred = svm_linear_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_linear_pred
        
        
        st.write(evaluate_model(svm_linear_pred, "SVM Linear"))
        
        
        st.write(" ")
        st.subheader("Poly Kernel")
        svm_poly_model = SVC(kernel='poly', degree=2)
        st.write("Training process...")
        svm_poly_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Poly Prediction with Testing dataset")
        svm_poly_pred = svm_poly_model.predict(X_test)
        st.write("Predictions from Linear Kernel Model for Testing Dataset: ")
        svm_poly_pred
        
        
        st.write(evaluate_model(svm_poly_pred, "SVM Poly"))
                 
        st.write(" ")
        st.subheader("RBF Kernel")
        svm_rbf_model = SVC(kernel='rbf')
        st.write("Training process...")
        svm_rbf_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM RBF Prediction with Testing dataset")
        svm_rbf_pred = svm_rbf_model.predict(X_test)
        st.write("Predictions from RBF Kernel Model for Testing Dataset: ")
        svm_rbf_pred
        
        
        st.write(evaluate_model(svm_rbf_pred, "SVM RBF"))
        
        
        st.write(" ")
        st.subheader("Sigmoid Kernel")
        svm_sigmoid_model = SVC(kernel='sigmoid')
        st.write("Training process...")
        svm_sigmoid_model.fit(X_train, y_train)
        
        st.write("Training Success...")
        
        st.write("SVM Sigmoid Prediction with Testing dataset")
        svm_sigmoid_pred = svm_sigmoid_model.predict(X_test)
        st.write("Predictions from Sigmoid Kernel Model for Testing Dataset: ")
        svm_sigmoid_pred
        
        
        st.write(evaluate_model(svm_sigmoid_pred, "SVM Sigmoid"))
        
    elif model == "K-Nearest Neighbors":
        
        neighbors = [10,50,100]
        
        for n in neighbors:
            st.subheader(f"K-Nearest Neighbors model (n_neighbors = {n})")
            knn_model = KNeighborsClassifier(n_neighbors= n)
            st.write("Training the Model...")
            knn_model.fit(X_train, y_train)

            st.write("Successfully Trained the model")
            knn_pred = knn_model.predict(X_test)
            st.write("Predicted result for Testing Dataset:")
            knn_pred
            
            st.write(evaluate_model(knn_pred, f"KNN model (n = {n})"))
    
    elif model == "Random Forest":
        
        st.subheader("Random Forest Salary Prediction model")
        
        n = [10, 50, 100, 150]
        for n in n:
            st.write("N_Estimator =",n)            

            rf_model = RandomForestClassifier(n_estimators= n, random_state=42)
            st.write("Training Process...")
            rf_model.fit(X_train, y_train)

            st.write("Training Success...")
            rf_pred = rf_model.predict(X_test)
            st.write("Predictions for Testing Dataset: ")
            rf_pred
            
            st.write(evaluate_model(rf_pred, f"Random Forest (n = {n})"))
            
     
    con = st.sidebar.selectbox("Select Summary", options = ["Select Summary", "Summary"])
    if con == "Summary":
        st.subheader("Conclusions from all Model: ")
        results_df = pd.DataFrame(results_dict).transpose()
        results_df.index.name = 'Model'
        results_df.reset_index(inplace=True)

        # Display the results table
        st.table(results_df)


