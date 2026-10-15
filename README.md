###### Employee Attrition Prediction System



A Python-based machine learning project that predicts employee attrition using employee-related data. The project also includes a FastAPI REST API for making predictions.



###### \--> About the Project



Employee attrition can be influenced by factors such as age, income, job role, distance from home, and other employee-related information.



This project uses employee data to train machine learning models and predict whether an employee is likely to leave the organization.



The project covers the complete flow from data preprocessing and model training to serving predictions through an API.



###### \--> Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- FastAPI

\- REST API

\- Git \& GitHub



###### \--> Machine Learning



The project uses the following machine learning algorithms:



\- Logistic Regression

\- Decision Tree

\- Random Forest



The models are evaluated using:



\- Accuracy

\- Precision

\- Recall

\- F1 Score

\- Confusion Matrix



###### \--> Project Workflow



1\. Load the employee dataset

2\. Clean and preprocess the data

3\. Convert categorical values into numerical values

4\. Prepare the features and target variable

5\. Split the data into training and testing sets

6\. Train the machine learning models

7\. Evaluate the models

8\. Save the trained model

9\. Use FastAPI to serve predictions through a REST API



###### \--> FastAPI REST API



The trained model is connected to a FastAPI application that provides REST API endpoints for checking the API status, making attrition predictions, and retrieving prediction results.



###### \--> Available Endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | `/` | Check the API status |

| GET | `/health` | Health check |

| POST | `/predict` | Predict employee attrition |

| GET | `/predictions` | Get prediction results |



FastAPI provides interactive API documentation through Swagger UI, which can be used to test the endpoints directly from the browser.



###### \--> Run the API



```bash

uvicorn app.main:app --reload--> Run the API



```bash

uvicorn app.main:app --reload

