
# Customer-Churn-Prediction-App

[My Streamlit app can be found here!](<https://customer-churn-prediction-app-789.streamlit.app>) 

![farrel-nobel-G9neENK1Z5I-unsplash](https://github.com/aswinram1997/Customer_Churn_Prediction/assets/102771069/8d866b16-2df3-48b2-a0eb-87e47beff7db)

## Project Overview
E-commerce companies struggle to predict customer churn, making ML essential.  The [Kaggle Dataset](<https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction?datasetId=1119908&sortBy=voteCount>) aids in creating a [Streamlit web app](<https://customer-churn-prediction-app-789.streamlit.app>) using object-oriented programming. This enables users to predict churn, understand reasons, and improve retention using both the churn prediction model and the dashboard included in the Web Application. The app compares five models - logistic regression, SVM, Random Forest, XGBoost, and DNN, integrating the chosen model in the Web application. It offers accurate predictions and structured code for seamless updates, enhancing e-commerce with churn insights. Through the app's prediction page, customer support teams can pinpoint individual cases that are at risk of churning. Simultaneously, the app's dashboard page enables the identification of larger trends and patterns in customer behavior. By harnessing both aspects, support teams can strategically allocate resources and make well-informed decisions. This approach not only minimizes churn but also elevates customer satisfaction while optimizing support efforts for maximum impact.

## Dataset Overview
The [Kaggle Dataset](<https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction?datasetId=1119908&sortBy=voteCount>) consists of 5630 rows of churn data. Within this dataset, a wealth of information related to customer behavior is meticulously documented. This diverse range of customer-centric data points provides a comprehensive and detailed perspective on the factors that contribute to churn within the context of the e-commerce company. 
- CustomerID: Unique customer ID
- Churn: Churn Flag
- Tenure: Tenure of customer in the organization
- PreferredLoginDevice: Preferred login device of the customer
- CityTier: City tier
- WarehouseToHome: Distance between the warehouse and the customer's home
- PreferredPaymentMode: Preferred payment method of the customer
- Gender: Gender of the customer
- HourSpendOnApp: Number of hours spent on the mobile application or website
- NumberOfDeviceRegistered: Total number of devices registered to the particular customer
- PreferredOrderCat: Preferred order category of the customer in the last month
- SatisfactionScore: Customer's satisfaction score for the service
- MaritalStatus: Marital status of the customer
- NumberOfAddress: Total number of addresses added by the customer
- Complain: Whether any complaint has been raised in the last month
- OrderAmountHikeFromlastYear: Percentage increase in order amount from the previous year
- CouponUsed: Total number of coupons used in the last month
- OrderCount: Total number of orders placed in the last month
- DaySinceLastOrder: Days since the last order by the customer
- CashbackAmount: Average cashback received in the last month

## Methodology
The project follows a specific workflow for insurance premium prediction using the provided dataset:

- Data Collection:The dataset containing insured data, including attributes and insurance charges, is collected as the initial step.

- Exploratory Data Analysis (EDA):EDA is conducted to gain insights into the dataset, identify patterns, and understand the relationships between attributes and insurance charges. This analysis provides valuable information for feature selection and modeling.

- Data Splitting:The preprocessed dataset is split into training, and testing sets. The training set is used to train the prediction models, the testing set is used for evaluation.

- Data Preprocessing:Data preprocessing involves several steps, including data cleaning, feature scaling, feature encoding, outlier removal, and handling imbalanced datasets which may involve creating new features or transforming existing ones to better represent the underlying relationships.

- Modeling, Evaluation, and Interpretation:Five prediction models, logistic regression, Support Vector Machine, Random Forest, Xtreme Gradient Boosting, and Deep Neural Network, are trained, evaluated, and interpreted with SHAP values using the training, and testing data. The performance of each model is assessed using the ROC-AUC score. The ANN is identified as the winning algorithm based on superior performance.

## Results
The results indicate that all models demonstrated reasonable generalization capabilities. However, the DNN model exhibited better overall performance in accurately predicting customer churn, as reflected by higher ROC-AUC scores across the train, and test sets. This suggests that the DNN model is not only accurate but also effectively captures the underlying patterns and relationships in the data, making it a preferred choice for customer churn prediction.

## Conclusion
The development of the Streamlit web app utilizing object-oriented programming for icustomer churn prediction and interpretation follows a specific methodology. This includes data collection, exploratory data analysis, data preprocessing, data splitting, modeling, and evaluation. By employing object-oriented programming, the project enhances code organization, reusability, and maintainability. The DNN model emerges as the preferred algorithm for insurance premium prediction based on its superior performance. The web app provides the e-commerce company a valuable tool for accurate churn prediction and interpretation, enabling optimal pricing strategies, risk management, and data-driven decision-making. 



