A portfolio of machine learning projects demonstrating end-to-end workflows, from data preparation and feature engineering to model training, evaluation, and insights. 🤖

### Project1 Name: RETAIL SALES FORECASTING AND PROFIT ANALYSIS SYSTEM. 

### 📌 Project Overview

- Analyzed 10,000 retail transactions to uncover revenue drivers across customer segments, cities, states, product categories, and shipping modes.
- Cleaned and prepared the dataset with feature engineering (e.g., time to deliver, ship month), label encoding, and outlier handling.
- Trained and tuned Linear Regression and XGBoost models, achieving ~65% R² accuracy.
- Identified discount as the most influential factor impacting profit margins.
- Visualized performance using residual plots, prediction error plots, learning curves, and feature importance charts.
- Findings provides actionable intelligence for retailers to balance discounts, shipping choices, and customer segmentation. 

### 📂 Repository Contents

- Project1.ipynb → Full Jupyter notebook (EDA → preprocessing → modeling → evaluation).
- Project1.pdf → Report summarizing methodology, analysis, and findings.

### 📊 Results

- Model Performance: Linear Regression → R² ≈ 0.62, consistent residual distribution.
- XGBoost → R² ≈ 0.65, reduced error after tuning.
- Key Insights: Discount emerged as the strongest predictor of profitability. Outliers in deposit/discount strongly affected model stability. Feature engineering improved model accuracy and interpretability.
- Visual Outputs:

- Residual Plot + Residual Distribution (Linear Regression). 
<img width="700" height="500" alt="Residual_Plot Distribution" src="https://github.com/user-attachments/assets/bebabd2a-369b-417d-8e22-ab757a184632" />

- RMSE Learning Curve (XGBoost).
<img width="700" height="400" alt="XGBoost_RMSE" src="https://github.com/user-attachments/assets/82c44d6f-ea4e-442b-8e78-08e049216d12" />

- Prediction Error Plot.
<img width="700" height="400" alt="Prediction_Error_Plot" src="https://github.com/user-attachments/assets/affe033d-d9aa-446a-9181-7aac6e6ef515" />

### 🔮 Future Scope

- Extend to multi-algorithm comparison (Random Forest, Gradient Boosting).
- Integrate external data (e.g., economic indicators, seasonal effects, market trends).
- Deploy as a Streamlit dashboard for interactive business users.

### 💻 Tech Stack 

• Python  • Pandas  • NumPy  • Matplotlib  • Seaborn  • Scikit-learn (Linear Regression)  • XGBoost

___________________________________________________________________________________________________________________________________________________________________________________________


### Project2 Name: NBA PLAYER PERFORMANCE PREDICTION.  

### 📌 Project Overview

- Analyzed 645k+ NBA player performance records from Kaggle to forecast player impact using machine learning.
- Preprocessed the dataset: dropped high-null columns, handled missing values, corrected datatypes (e.g., minutes → float), and engineered a new target variable (Modified Plus-Minus Score).
- Reduced dataset to ~516k rows × 27 columns after cleaning and feature optimization.
- Conducted exploratory data analysis with bar plots, scatter plots, and heatmaps to uncover team-level performance patterns and player statistics.
- Built a Logistic Regression model (62% accuracy) to classify player impact; compared with K-Nearest Neighbors (~56% accuracy).
- Evaluated models using confusion matrix, classification report, and feature weights to highlight the most influential performance metrics.
- Developed a career stats function, allowing input of a player’s name to compute aggregated performance across seasons.
- Results help coaches, analysts, and teams make data-driven decisions on player contributions.

### 📂 Repository Contents

- Project2.ipynb → Jupyter notebook with full preprocessing, feature engineering, EDA, and modeling workflow. 

### 📊 Results

- Dataset reduced from 645k+ to ~516k rows after cleaning and feature optimization.
- Logistic Regression achieved ~62% accuracy on training and test sets.
- K-Nearest Neighbors (KNN) reached ~56% accuracy (confirming Logistic Regression as the stronger model).
- Evaluation metrics:
- Classification Report → precision, recall, F1-score across classes.
<img width="412" height="174" alt="ClassificationReport" src="https://github.com/user-attachments/assets/61300ffd-9668-45fa-8e51-f162c14de6e8" />

- Confusion Matrix → distribution of correct vs incorrect predictions.
<img width="450" height="400" alt="ConfusionMatrix" src="https://github.com/user-attachments/assets/259d2d2b-ffa6-4d96-b205-bbbadb1a7d83" />

### 🔮 Future Scope

- Incorporate contextual variables like home/away games, opponent defense strength, or player fatigue for richer predictions. 
- Experiment with ensemble methods (Random Forest, XGBoost) and deep learning models (RNNs for time-series game data).

### 💻 Tech Stack 

• Python  • Pandas • Matplotlib  • Seaborn  • Scikit-learn (Logistic Regression, KNN)  • Pickle
