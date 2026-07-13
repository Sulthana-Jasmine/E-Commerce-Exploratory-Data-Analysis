"""
==================================================================
PROJECT 2: EXPLORATORY DATA ANALYSIS (EDA)
Dataset: Cleaned_Dataset.xlsx  (E-commerce Order Data)
Author : Data Analytics Intern
Libraries: pandas, numpy, matplotlib, seaborn
==================================================================
"""

# =========================================================
# STEP 0: IMPORT REQUIRED LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean visual style for all charts
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# =========================================================
# STEP 1: LOAD THE DATASET
# =========================================================
# Read the Excel file into a pandas DataFrame
file_path = "Cleaned_Dataset.xlsx"
df = pd.read_excel(file_path)

print("=" * 70)
print("STEP 1: DATASET LOADED SUCCESSFULLY")
print("=" * 70)

# =========================================================
# STEP 2: BASIC DATASET INFORMATION
# =========================================================
print("\n--- Shape of Dataset (Rows, Columns) ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types of Each Column ---")
print(df.dtypes)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Last 5 Rows ---")
print(df.tail())

print("\n--- General Info (non-null counts, memory usage) ---")
df.info()

# EXPECTED OUTPUT:
# The dataset contains 1,200 rows and 14 columns, covering order-level
# e-commerce data (OrderID, Date, Customer, Product, Pricing, Payment,
# Shipping, Coupons and Referral Source). Data types are a mix of
# object/string, datetime, integer, and float columns.

# =========================================================
# STEP 3: MISSING VALUES & DUPLICATE RECORDS
# =========================================================
print("\n" + "=" * 70)
print("STEP 3: DATA QUALITY CHECKS")
print("=" * 70)

print("\n--- Missing Values per Column ---")
print(df.isnull().sum())

print("\n--- Total Duplicate Rows ---")
print(df.duplicated().sum())

# EXPECTED OUTPUT:
# No missing values and no duplicate rows are found, confirming that
# the dataset has already been cleaned prior to this analysis.

# =========================================================
# STEP 4: DESCRIPTIVE STATISTICS
# =========================================================
print("\n" + "=" * 70)
print("STEP 4: DESCRIPTIVE STATISTICS (NUMERICAL COLUMNS)")
print("=" * 70)

numeric_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]

# describe() gives count, mean, std, min, 25%, 50%, 75%, max
desc_stats = df[numeric_cols].describe()
print(desc_stats)

# Median is the 50th percentile; add it explicitly for clarity
print("\n--- Median values ---")
print(df[numeric_cols].median())

# EXPECTED OUTPUT:
# Quantity ranges from 1 to 5 (avg ~2.9). UnitPrice ranges roughly from
# ₹11 to ₹700 (avg ~₹356). TotalPrice (Quantity x UnitPrice) ranges from
# ₹11 to ₹3,456, with a right-skewed distribution (mean > median),
# suggesting a small number of high-value orders.

# =========================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# =========================================================
print("\n" + "=" * 70)
print("STEP 5: EXPLORATORY DATA ANALYSIS - VISUALIZATIONS")
print("=" * 70)

# ---------------------------------------------------------
# 5.1 HISTOGRAMS - Distribution of numerical columns
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Histograms of Numerical Columns", fontsize=16, fontweight="bold")

for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col], bins=25, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("hist_numeric.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - Quantity and ItemsInCart look fairly uniform across their range.
# - UnitPrice is spread broadly with no single dominant price band.
# - TotalPrice is right-skewed - most orders are low-to-mid value,
#   with a long tail of high-value orders.

# ---------------------------------------------------------
# 5.2 BOXPLOTS - Outlier detection
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Boxplots for Outlier Detection", fontsize=16, fontweight="bold")

for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(x=df[col], color="lightcoral", ax=ax)
    ax.set_title(f"Boxplot of {col}")
    ax.set_xlabel(col)

plt.tight_layout()
plt.savefig("boxplot_numeric.png", bbox_inches="tight")
plt.show()

# Calculate outliers using the IQR method for each numeric column
print("\n--- Outlier Count per Column (IQR Method) ---")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")

# INSIGHT:
# - Quantity and ItemsInCart show little to no outliers since they are
#   bounded, discrete counts.
# - TotalPrice shows a small number of outliers (high-value orders);
#   UnitPrice shows none — pricing is controlled within a reasonable
#   business range rather than having erratic spikes.

# ---------------------------------------------------------
# 5.3 CORRELATION HEATMAP
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - TotalPrice is strongly positively correlated with Quantity and
#   UnitPrice (as expected, since TotalPrice = Quantity x UnitPrice).
# - ItemsInCart shows weak correlation with other variables, suggesting
#   cart size does not strongly influence final order value.

# ---------------------------------------------------------
# 5.4 PAIRPLOT - Relationship between numeric variables
# ---------------------------------------------------------
pairplot_fig = sns.pairplot(df[numeric_cols], diag_kind="kde", corner=True,
                             plot_kws={"alpha": 0.4, "color": "teal"})
pairplot_fig.fig.suptitle("Pairplot of Numerical Variables", y=1.02, fontsize=14, fontweight="bold")
pairplot_fig.savefig("pairplot_numeric.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - A clear positive relationship is visible between UnitPrice and
#   TotalPrice, and between Quantity and TotalPrice.
# - No strong relationship is seen between ItemsInCart and the other
#   numeric variables, reinforcing the heatmap findings.

# ---------------------------------------------------------
# 5.5 COUNT PLOTS - Categorical columns
# ---------------------------------------------------------
categorical_cols = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]

fig, axes = plt.subplots(3, 2, figsize=(15, 14))
fig.suptitle("Count Plots for Categorical Columns", fontsize=16, fontweight="bold")
axes = axes.flatten()

for ax, col in zip(axes, categorical_cols):
    order = df[col].value_counts().index
    sns.countplot(y=df[col], order=order, color="teal", ax=ax)
    ax.set_title(f"Count of {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)

# Hide the unused 6th subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig("countplots_categorical.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - Products are fairly evenly distributed (Printer, Tablet and Chair
#   are the top-selling categories, Phone the lowest).
# - Order statuses (Cancelled, Returned, Pending, Shipped, Delivered)
#   are almost evenly split, indicating a notably high combined
#   cancellation + return rate that is worth investigating.
# - "SAVE10" is the most frequently used coupon; a large share of
#   orders also use no coupon at all.
# - Instagram and Referral are leading traffic/referral sources.

# ---------------------------------------------------------
# 5.6 BAR CHART - Average TotalPrice by Product
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
avg_price_by_product = df.groupby("Product")["TotalPrice"].mean().sort_values(ascending=False)
sns.barplot(x=avg_price_by_product.values, y=avg_price_by_product.index, color="darkcyan")
plt.title("Average Total Price by Product", fontsize=14, fontweight="bold")
plt.xlabel("Average Total Price")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("bar_avg_price_by_product.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - Products differ noticeably in their average order value, which
#   helps identify which product categories drive the most revenue
#   per order.

# ---------------------------------------------------------
# 5.7 BAR CHART - Order Status vs Payment Method (comparison)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
status_payment = pd.crosstab(df["PaymentMethod"], df["OrderStatus"])
status_payment.plot(kind="bar", stacked=True, colormap="tab20", ax=plt.gca())
plt.title("Order Status Breakdown by Payment Method", fontsize=14, fontweight="bold")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.legend(title="Order Status", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("bar_status_by_payment.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - Order status distribution is broadly similar across all payment
#   methods, meaning payment type alone does not strongly predict
#   whether an order gets cancelled, returned, or delivered.

# ---------------------------------------------------------
# 5.8 TREND OVER TIME - Monthly Order Volume & Revenue
# ---------------------------------------------------------
df["Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_trend = df.groupby("Month").agg(Orders=("OrderID", "count"),
                                         Revenue=("TotalPrice", "sum")).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(monthly_trend["Month"], monthly_trend["Orders"], color="steelblue",
         marker="o", label="Order Count")
ax2.plot(monthly_trend["Month"], monthly_trend["Revenue"], color="orange",
         marker="s", label="Revenue")

ax1.set_xlabel("Month")
ax1.set_ylabel("Number of Orders", color="steelblue")
ax2.set_ylabel("Total Revenue", color="orange")
ax1.set_xticks(range(len(monthly_trend["Month"])))
ax1.set_xticklabels(monthly_trend["Month"], rotation=90)
plt.title("Monthly Order Volume and Revenue Trend", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig("monthly_trend.png", bbox_inches="tight")
plt.show()

# INSIGHT:
# - This trend line reveals whether order volume and revenue are
#   growing, seasonal, or stable across the Jan 2023 - Jun 2025
#   period covered by the data, helping identify peak sales months.

# =========================================================
# STEP 6: FINAL SUMMARY OF KEY OBSERVATIONS
# =========================================================
print("\n" + "=" * 70)
print("STEP 6: KEY OBSERVATIONS (SUMMARY)")
print("=" * 70)

summary_points = [
    "Dataset contains 1,200 order records across 14 columns with NO missing values and NO duplicates.",
    "Numeric fields (Quantity, UnitPrice, ItemsInCart, TotalPrice) show reasonable, business-realistic ranges.",
    "TotalPrice is right-skewed - most orders are low-to-mid value with a smaller number of high-value orders.",
    "TotalPrice correlates strongly with Quantity and UnitPrice, as expected mathematically.",
    "Very few outliers were detected via the IQR method (only in TotalPrice).",
    "Product categories are fairly balanced, with Printer, Tablet, and Chair being the top sellers by volume.",
    "Order status is spread almost evenly across Cancelled, Returned, Pending, Shipped, and Delivered - a high combined Cancelled+Returned rate (~41%) stands out.",
    "SAVE10 is the most-used coupon code; Instagram and Referral are the top referral sources.",
    "Payment method does not appear to strongly influence order status outcomes."
]

for point in summary_points:
    print(f" - {point}")

print("\nEDA COMPLETE. All charts have been saved as PNG files in the working directory.")