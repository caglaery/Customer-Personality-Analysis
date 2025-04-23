# Marketing Campaign Data Analysis

This repository contains a comprehensive analysis of a marketing campaign dataset, including data cleaning, exploratory data analysis (EDA), RFM (Recency, Frequency, Monetary) analysis, customer segmentation, and CLTV (Customer Lifetime Value) estimation using **BG-NBD** (Beta Geo Model) and **Gamma-Gamma Model**.

## **Project Overview**

The project provides insights into customer behavior based on their interaction with a marketing campaign. The primary objective is to determine customer segments, predict CLTV (Customer Lifetime Value), and analyze factors such as income, frequency of purchases, and response to different campaigns.

### **Libraries Used**

```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import matplotlib.ticker as mtick
from sklearn.preprocessing import MinMaxScaler
```

## **Data Cleaning and Preprocessing**

1. **Data Import:**

   The data is loaded using `pandas` from a CSV file, with tab (`\t`) as the separator.

   ```python
   df = pd.read_csv("marketing_campaign.csv", sep="\t")
   ```

2. **Date Transformation:**

   The date column `Dt_Customer` is converted into `datetime` format to facilitate time-based analysis.

   ```python
   df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
   ```

3. **Outlier Detection and Removal:**

   Outliers in variables like `Income` are detected using the IQR (Interquartile Range) method and replaced with threshold values.

   ```python
   def outlier_thresholds(dataframe, variable):
       q1 = dataframe[variable].quantile(0.25)
       q3 = dataframe[variable].quantile(0.75)
       iqr = q3 - q1
       lower = q1 - 1.5 * iqr
       upper = q3 + 1.5 * iqr
       return lower, upper
   ```

   ```python
   def replace_wth_thresholds(dataframe, variable):
       lower, upper = outlier_thresholds(dataframe, variable)
       dataframe.loc[(dataframe[variable] < lower), variable] = round(lower, 0)
       dataframe.loc[(dataframe[variable] > upper), variable] = round(upper, 0)
   ```

4. **Customer Age Calculation:**

   A new column `Age` is created based on the `Year_Birth` column and the current date.

   ```python
   df["Age"] = today_date.year - df["Year_Birth"]
   ```

## **Exploratory Data Analysis (EDA)**

### 1. **Visualizing the `Income` Distribution:**

   Boxplot to detect outliers in the `Income` column:

   ```python
   sns.boxplot(y=df["Income"], fliersize=12, linewidth=0.5, color="lightgreen")
   plt.title('Income Distribution')
   plt.show()
   ```

### 2. **Recency vs. Age Analysis:**

   A line plot is used to visualize the relationship between `Age` and `Recency`.

   ```python
   sns.lineplot(data=df, x="Age", y="Recency")
   plt.title('Recency vs Age')
   plt.show()
   ```

### 3. **Customer Segmentation by Age:**

   A bar plot shows the distribution of complaints by `Age_Cat`.

   ```python
   df_percent = df.groupby("Age_Cat")["Complain"].value_counts(normalize=True).rename("Percentage").reset_index()
   sns.barplot(data=df_percent, x="Age_Cat", y="Percentage", hue="Complain", palette="Set2")
   plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
   plt.title('Complaint Percentage by Age Category')
   plt.show()
   ```

### 4. **Total Purchases vs. Monetary Value:**

   Heatmap to explore correlations between `Total_Acceptance`, `Monetary`, and other key variables.

   ```python
   sns.heatmap(df[["Total_Acceptance", "TotalMnt", "Recency", "Income", "Monetary"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
   plt.title('Correlation Matrix')
   plt.show()
   ```

## **Customer Segmentation and RFM Analysis**

### 1. **Customer Segmentation:**

   The `RF_Score` is calculated by combining the Recency and Frequency scores, and customers are categorized into segments.

   ```python
   df["RF_Score"] = (df["Recency_Score"].astype(str)) + (df["Frequency_Score"].astype(str))
   ```

   Segments are defined using the following mapping:

   ```python
   seg_map = {
       r'[1-2][1-2]': "hibernating",
       r'[1-2][3-4]': "at_Risk",
       r'[1-2]5': "cant_loose",
       r'3[1-2]': "about_to_sleep",
       r'33': "need_attention",
       r'[3-4][4-5]': "loyal_customers",
       r'41': "promising",
       r'51': "new_customers",
       r'[4-5][2-3]': "potential_loyalists",
       r'5[4-5]': "champions"
   }
   df["Segment"] = df["RF_Score"].replace(seg_map, regex=True)
   ```

### 2. **Heatmap of Campaign Acceptance Status by Segment:**

   A heatmap is used to visualize how different segments have responded to each campaign.

   ```python
   pivot_df = df.pivot_table(index='Segment', values=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], aggfunc='sum')
   plt.figure(figsize=(10, 7))
   sns.heatmap(pivot_df, annot=True, fmt="d", cmap="YlOrRd", cbar=True)
   plt.title('Campaign Acceptance Status by Segment')
   plt.show()
   ```

## **CLTV Estimation with BG-NBD and Gamma-Gamma**

### 1. **BG-NBD Model:**

   The **Beta Geo Model** is used to predict the expected number of future purchases.

   ```python
   bgf = BetaGeoFitter(penalizer_coef=0.01)
   bgf.fit(cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"])
   ```

   Predicting the number of purchases in the next 3, 6, and 12 months:

   ```python
   bgf.conditional_expected_number_of_purchases_up_to_time(4, cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"]).sort_values(ascending=False).head(10)
   ```

### 2. **Gamma-Gamma Model:**

   The **Gamma-Gamma Model** estimates the average profit that customers will bring.

   ```python
   ggf = GammaGammaFitter(penalizer_coef=0.01)
   ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])
   ```

   Predicting the expected average profit for the top 10 customers:

   ```python
   cltv_df["exp_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"], cltv_df["Monetary"])
   ```

### 3. **CLTV Calculation:**

   The CLTV for each customer is calculated using both the BG-NBD and Gamma-Gamma models.

   ```python
   cltv = ggf.customer_lifetime_value(bgf, cltv_df["Frequency"], cltv_df["Recency"], cltv_df["T"], cltv_df["Monetary"], time=12, freq="W", discount_rate=0.1)
   ```

## **Visualizing Customer Lifetime Value (CLTV)**

### 1. **Scatter Plot of CLTV Estimates:**

   A scatter plot compares the predicted CLTV values to the actual calculated CLTV.

   ```python
   sns.scatterplot(data=cltv, x="CLTV", y="CLTV_form")
   plt.title('CLTV Comparison')
   plt.show()
   ```

### 2. **Correlation Heatmap of CLTV Variables:**

   A heatmap is used to show the correlation between the actual and predicted CLTV values.

   ```python
   corr_cltv = cltv[["CLTV", "CLTV_form"]].corr()
   sns.heatmap(corr_cltv, annot=True, cmap="coolwarm", fmt=".2f")
   plt.title("CLTV vs CLTV_form Correlation")
   plt.show()
   ```

---

## **Conclusion**

This analysis provides actionable insights into customer behavior based on their interaction with marketing campaigns, and it allows businesses to identify high-value customers, forecast future sales, and improve marketing strategies.
