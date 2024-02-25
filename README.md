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
<p>This section should detail our data preprocessing steps, feature engineering techniques, model selection criteria, evaluation metrics, and any cross-validation strategies employed.</p>

<h2>Results and Discussion</h2>
<p>This section should let us summarize the key findings, model performance metrics, insights gained from the clustering analysis, and any interesting patterns or anomalies uncovered during the analysis.</p>

<h2>Conclusions and Future Work</h2>
<p>Reflect on the project's impact on the business, potential improvements, and areas for future exploration, such as integrating more data sources or experimenting with advanced modeling techniques.</p>


<h2>Team Members</h2>
<ul>
<li><b>Ge Gao</b> - Data Scientist</li>
<li><b>Hongyi Zhan</b> - Data Scientist</li>
<li><b>Kelly Kao</b> - Data Scientist</li>
<li><b>Ko-Jen Wang</b> - Business Analyst</li>
<li><b>Rebecca Zhang</b> - Product Manager</li>
<li><b>Yanhuan Huang</b> - Data Analyst</li>


</ul>



