<h1>DataCo Smart Supply Chain Analysis</h1>

<h2>Executive Summary</h2>
<p>This project focuses on leveraging the DataCo Smart Supply Chain dataset from Kaggle to build predictive models that serve critical business functions: forecasting monthly demand, detecting fraudulent orders, and clustering orders to identify common traits in fraud cases. Through sophisticated data analysis and machine learning techniques, this initiative aims to enhance operational efficiency, mitigate risks, and drive strategic decision-making in supply chain management.</p>

<h2>Dataset Overview</h2>
<p>The dataset comprises supply chain data from DataCo Global, including transaction records, customer data, product information, and logistics. Key features include order IDs, product categories, quantities ordered, customer locations, payment methods, and more. This rich dataset provides a comprehensive view of the company's operations, serving as a foundation for our predictive models.</p>
<p>Key Dataset Features:</p>
<ul>
<li><b>Orders:</b> Contains over 180,000 unique order records, enabling detailed purchase pattern analysis and demand forecasting.</li>
<li><b>Unique Customers:</b> Features a diverse customer base with more than 30,000 unique profiles, ideal for customer segmentation and personalized strategy development.</li>
<li><b>Products:</b> Includes details on over 10,000 unique products, offering insights into inventory needs and product popularity.</li>
<li><b>Geographical Coverage:</b> The data spans multiple countries, allowing for global supply chain insights and regional analyses.</li>
<li><b>Fraud Indicators:</b> Contains markers for potential fraud, aiding in the creation of effective fraud detection models.</li>
<li><b>Rich Details:</b> Offers comprehensive information on customer demographics and order specifics such as shipping methods and payment channels.</li>
</ul>

<h2>Models</h2>
<h3>1. Monthly Demand Prediction</h3>
<p>This model aims to predict the monthly demand for products in the supply chain. By understanding demand patterns, the company can optimize inventory levels, reduce holding costs, and improve customer satisfaction.
</p>
<ul>
<li><b>Features Used:</b> Historical sales data, product categories, time-series features.</li>
<li><b>Algorithms Considered:</b> LSTM, Linear Regression, Gradient Boosting, Random Forest.</li></ul>

<h3>2. Fraud Order Detection Prediction</h3>
<p>This model focuses on identifying potentially fraudulent transactions. Early detection of fraud can significantly reduce financial losses and protect the company's reputation.
</p>
<ul>
<li><b>Features Used:</b> Payment methods, order amounts, customer behavior patterns.</li>
<li><b>Algorithms Considered:</b> Logistic Regression, Random Forest, Gradient Boosting.</li></ul>

<h3>3. Fraud Order Traits Clustering</h3>
<p>Beyond detection, understanding common traits among fraudulent orders can help in developing more effective prevention strategies. This model clusters fraudulent orders to uncover these traits. While fraudsters constantly evolve their strategies to bypass detection mechanisms, the model can adapt to new patterns by updating the model with new data.
</p>
<ul>
<li><b>Features Used:</b> Attributes of orders flagged as fraudulent.</li>
<li><b>Algorithms Considered:</b> K-Means, DBSCAN, Agglomerative Clustering.</li></ul>


<h2>Methodology</h2>
<p>As dealing with large dataset (more than 180K rows and more 40 columns), thorough preprocessing was conducted both in EDA stage and model development stage to enusre robust model outputs. EDA involved preprocessing steps like removing columns with distinct values and reconstruct order datetime columns, the cleaned dataset was exported as csv to aid model development steps. EDA looked at numerical variables and categorical variables separately, also contained OLS regression report for total profit per order, order counts through time, and fraud orders analysis, paving the way for model development.</p>
<p>For Demand Prediction, the models relied on aggregating variables such as sales, shipping days, and product price to predict monthly order quantities. After preprocessing the data, Random Forest algorithm is utilized for feature selection. This is followed by tuning several models, including Long Short-Term Memory (LSTM) networks, Random Forest (RF), Gradient Boosting (GB), and Logistic Regression (LR), to optimize predictions based on MAE and MSE comparison.</p>
<p>In the Fraud Detection, after preprocessing which includes feature engineering and label encoding, the imbalance in the dataset is addressed by oversampling fraudulent orders. Feature selection is then carried out through Recursive Feature Elimination (RFE). The model tuning phase employs ensemble methods—specifically a voting classifier comprising Logistic Regression, Random Forest, and Support Vector Machine (SVM)—as well as boosting algorithms like XGBoost and LightGBM to accurately identify fraudulent activities.</p>
<p>Fraud Order Clustering is another aspect looking at fraudulent orders. Preprocessing steps such as filtering and standardization are applied before determining the optimal number of clusters by calculating inertia. KMeans clustering was then used to group the data, along with an inverse transformation of the cluster centroids to visualize each cluster's features. Additionally, the DBSCAN algorithm is applied to obtain 6 clusters and identify noise within the dataset, enhancing the robustness of fraud identification.</p>

<h2>Results and Discussion</h2>
<p>In Demand Prediction, Gradient Boosting emerged as the superior model among four candidates, achieving a mean squared error of 9.05, indicating its effectiveness in forecasting product demand across various categories and regions, thereby facilitating inventory management. For Fraud Detection, the LightGBM model stood out with an impressive accuracy of 0.99 and precision of 0.65, outperforming other models. Notable features for detecting fraud included the type of transaction, late delivery risk, shipping mode, and order month. This detection is crucial in mitigating financial losses and refining demand prediction models.</p>


<h2>Team Members</h2>
<ul>
<li><b>Ge Gao</b> - Data Scientist</li>
<li><b>Hongyi Zhan</b> - Data Scientist</li>
<li><b>Kelly Kao</b> - Data Scientist</li>
<li><b>Ko-Jen Wang</b> - Business Analyst</li>
<li><b>Rebecca Zhang</b> - Product Manager</li>
<li><b>Yanhuan Huang</b> - Data Analyst</li>


</ul>



