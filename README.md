# da3assignment
# Model Comparison: Fit and Performance

## Introduction

I built 8 predictive models using data for **Berlin Q4** to assess their effectiveness in predicting price. The dataset initially contained **14,000 observations**, but after **removing missing price values**, I was left with **9,000 observations**. I then **cleaned the data, created the necessary variables, and proceeded with the predictions**. These models include **OLS, Lasso, Lasso with GridSearchCV, Random Forest, CART, GBM Boosting, MLP Regressor, and CatBoost Regressor**. The goal was to compare their fit and generalization performance using **Root Mean Squared Error (RMSE)** for both the training and holdout datasets.

## Results

The table below summarizes the RMSE scores for each model:

| Model                  | Training RMSE | Holdout RMSE |
|------------------------|--------------|--------------|
| OLS                   | 98.75        | 100.77       |
| Lasso                 | 90.45        | 104.39       |
| Lasso (GridSearchCV)  | 104.91       | 80.26        |
| Random Forest         | 99.82        | 72.13        |
| CART                  | 71.48        | 95.46        |
| GBM Boosting          | 49.40        | 96.05        |
| MLP Regressor         | 104.97       | 73.58        |
| CatBoost Regressor    | 63.41        | 69.15        |

### Performance Discussion

- **OLS and Lasso** perform relatively bad, with high RMSE values, more than 100 for holdout data.
- **Lasso with GridSearchCV** has a high training RMSE but achieves a strong holdout RMSE, maybe because of better generalization.
- **Random Forest** significantly improves performance over linear models, achieving a holdout RMSE of 72.13.
- **CART (Decision Tree)** overfits the data, performing well in training but poorly on the holdout set.
- **GBM Boosting** works effectively but fails to generalize it seems, as it has a high holdout RMSE.
- **MLP Regressor** behaves similarly to Lasso with GridSearchCV, showing a high training RMSE but a good holdout performance.
- **CatBoost Regressor** emerges as the best model, achieving the lowest holdout RMSE of **69.14**, making it the most effective model.

### Key Observations
1. **Tree-based models (Random Forest, CART, Boosting, CatBoost) outperform linear models (OLS, Lasso).**
2. **Boosting models, while powerful, risk overfitting** (e.g., GBM has a low training RMSE but high holdout RMSE).
3. **CatBoost provides the best trade-off between training and holdout performance, making it the best model for prediction, at list for Berlin.**

## Feature Importance Analysis: Random Forest vs. CatBoost

### Feature Importance Overview

We compare the top most important features in **Random Forest** and **CatBoost**. Feature importance scores indicate which variables have the greatest influence on predicting rental prices.

### Top Most Important Features

#### **Random Forest Feature Importance**
1. **Number of accommodates** – 6.3%  
2. **Number of bathrooms** – 5.6%  
3. **Room type** – 4.5%  
4. **Property type** – 4.1%  
5. **Number of bedrooms** – 3.4%  
6. **Neighbourhood** – 2.8%  
7. **Review scores rating** – 1.8%  
8. **Number of beds** – 1.6%  
9. **Host acceptance rate** – 0.8%  
10. **Number of reviews** – 0.7%  
11. **Number of days since the last review** – 0.6%  

#### **CatBoost Feature Importance**
1. **Number of accommodates** – 16.25  
2. **Number of days since the last review** – 13.06  
3. **Neighbourhood** – 12.48  
4. **Property type** – 9.33  
5. **Number of bedrooms** – 8.95  
6. **Room type** – 8.41  
7. **Number of bathrooms** – 7.98  
8. **Host acceptance rate** – 6.08  
9. **Review scores rating** – 4.66  
10. **Number of beds** – 3.12  
11. **Host response time** – 2.26  
12. **Host response rate** – 1.81  

### Comparison and Discussion

- **Key Variables:** In both models **number of accommodates**, **property type**, **room type**, and **number of bedrooms** play a major role in determining price.
- **Ranking Differences:**
- - Random Forest assigns **less importance** to top variables compared to CatBoost because it splits data using a series of independent decision trees, averaging their results, which distributes importance across many features. In contrast, CatBoost uses gradient boosting, which sequentially refines errors, allowing it to assign higher weight to the most impactful variables. Additionally, CatBoost handles categorical variables more efficiently, capturing deeper interactions that may not be as emphasized in Random Forest.
- **Neighbourhood** is much more important in CatBoost (12.48) than in Random Forest (2.8%), likely due to its ability to capture non-linear interactions.
  - **Number of days since the last review** is a major feature in CatBoost (13.06) but is nearly **absent in Random Forest**, showing how boosting models capture **recency effects** better.
- **Model Strengths:**  
  - **Random Forest** relies more on **physical attributes** (bathrooms, bedrooms, beds).  
  - **CatBoost** captures **temporal trends** (days since last review) and **categorical effects** (neighbourhoods, host-related features).


### Conclusion

CatBoost appears to provide **a more refined feature ranking** that incorporates **both temporal and categorical influences**, whereas Random Forest focuses more on **structural aspects of the rental properties**. This suggests that **CatBoost may generalize better** in dynamic environments where recency and location play a critical role.

# Part II. Validity

## Evaluating Models on "Live" Datasets

To test the validity of the models, I applied them to two different datasets:

1. **Berlin Q3 Data** (used instead of a later date since I originally used the latest data)
2. **Münich Data** (a different city in the same country with at least 3,000 observations)

The goal was to examine how well the models generalize to different datasets.

### A. Berlin Q3 Data

Below are the results when applying the models to Berlin Q3:

| Model                      | Training RMSE | Holdout RMSE |
|----------------------------|--------------|--------------|
| OLS                        | 114.106198   | 121.624042   |
| Lasso                      | 100.555182   | 133.028446   |
| Lasso (GridSearchCV)       | 116.638997   | 100.776692   |
| Random Forest              | 110.247099   | 101.036873   |
| CART                       |  69.522562   | 132.179169   |
| GBM Boosting               |  88.259464   | 124.408528   |
| MLP Regressor              |  97.573471   |  98.846540   |
| CatBoost Regressor         |  76.376189   |  98.506254   |

The results show that **MLP and CatBoost performed the best in terms of holdout RMSE**, while models like **CART, OLS and Lasso** struggled with higher RMSE values for the Holdout data.

---

### B. Münich Data

Below are the results when applying the models to the Münich dataset:

| Model                      | Training RMSE | Holdout RMSE |
|----------------------------|--------------|--------------|
| OLS                        | 92.568161    | 91.396249    |
| Lasso                      | 91.707899    | 92.472488    |
| Lasso (GridSearchCV)       | 85.624930    | 105.570629   |
| Random Forest              | 76.708829    | 103.253418   |
| CART                       | 20.373133    | 119.529932   |
| GBM Boosting               | 74.364406    | 103.885723   |
| MLP Regressor              | 87.778648    | 113.229255   |
| CatBoost Regressor         | 73.478226    | 86.215283    |

In Münich, **CatBoost again performed the best on the holdout set**, followed by **OLS and Lasso**. **Random Forest, MLP and GBM had higher errors than in Berlin**, maybe because of some potential issues with generalization.

---

## Discussion

- **CatBoost consistently performed well across datasets**, achieving the lowest holdout RMSE in **both Berlin Q3 and Münich**.
- **Random Forest performed well in Berlin Q4 but struggled in Münich**, suggesting sensitivity to different city characteristics.
- **Lasso (GridSearchCV) and MLP performed well in Berlin datasets but not in Münich, whereas Lasso alone had better holdout RMSE in Münich.**  
  This suggests that **Lasso's performance varies depending on the dataset**, and in some cases, **hyperparameter tuning (GridSearchCV) does not always improve generalization**.
- **Overall, the RMSE values for Berlin Q3 were worse than Berlin Q4.**  
  This could be due to changes in data distribution between the two periods, **indicating that models trained on Q4 data might not generalize as well to earlier datasets**.
- **OLS and Lasso performed surprisingly well in Münich, despite being simple linear models.**  
  This might indicate that **the Münich dataset had a more linear relationship between features and price**, making simpler models more effective.
- **Lasso (GridSearchCV) in Münich was the only case where holdout RMSE was worse than training RMSE.**  
  This suggests that **the hyperparameters selected by GridSearchCV might have led to overfitting**, making the model **less effective on unseen data**.
- **CART continues to show poor performance as a predictive model.**  
  Despite having a **very low training RMSE**, it **consistently has high holdout RMSE**, confirming that **CART overfits heavily and does not generalize well**.
- **Boosting models (GBM and CatBoost) provided strong results**, but **GBM was less stable compared to CatBoost**.

These results highlight the **importance of testing models on different datasets** to ensure they generalize well.

## Conclusion

In this project, I built **eight different predictive models** to analyze Airbnb price predictions using data from **Berlin (Q4 & Q3) and Münich**. The models included **OLS, Lasso, Lasso (GridSearchCV), Random Forest, CART, GBM Boosting, MLP Regressor, and CatBoost Regressor**.


### Final Thoughts:

This analysis **reinforces the importance of model selection** based on the dataset characteristics. While **tree-based methods (Random Forest, CatBoost) are generally more robust**, **Lasso and OLS can perform surprisingly well when relationships are more linear**. 

Future improvements could include:
- Exploring **more hyperparameter tuning** for boosting models.
- Applying **ensemble methods** that combine multiple models for better stability.
- Investigating **feature engineering techniques** to improve predictive power.

By ensuring a **reproducible and structured workflow**, this project provides valuable insights into predictive modeling for Airbnb pricing.

**AI was used for code generation, language editing and formatting**
