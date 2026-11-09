
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Path to your dataset (falls back to the same folder as this script)
DATA_PATH = r"C:\Users\Krishnamoorthe\OneDrive\Desktop\intern\WA_Fn-UseC_-HR-Employee-Attrition.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def heading(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)



# TASK 1. UNDERSTAND THE DATASET AND CREATE A DATA DICTIONARY

heading("TASK 1: UNDERSTAND THE DATASET + DATA DICTIONARY")

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()          # work on a copy; keep the raw data untouched

print("Shape (rows, columns):", df_raw.shape)
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nColumn data types:")
print(df_raw.dtypes)

descriptions = {
    "Age": "Employee age in years",
    "Attrition": "Whether the employee left the company (Yes/No) - TARGET",
    "BusinessTravel": "Frequency of business travel",
    "DailyRate": "Daily billing rate",
    "Department": "Department (Sales, Research & Development, Human Resources)",
    "DistanceFromHome": "Distance from home to workplace",
    "Education": "Education level (1 Below College, 2 College, 3 Bachelor, 4 Master, 5 Doctor)",
    "EducationField": "Field of study",
    "EmployeeCount": "Always 1 (constant)",
    "EmployeeNumber": "Unique employee ID",
    "EnvironmentSatisfaction": "Satisfaction with work environment (1 Low - 4 Very High)",
    "Gender": "Male / Female",
    "HourlyRate": "Hourly billing rate",
    "JobInvolvement": "Job involvement (1 Low - 4 Very High)",
    "JobLevel": "Job level (1 entry - 5 senior)",
    "JobRole": "Job role / title",
    "JobSatisfaction": "Job satisfaction (1 Low - 4 Very High)",
    "MaritalStatus": "Single / Married / Divorced",
    "MonthlyIncome": "Monthly salary",
    "MonthlyRate": "Monthly billing rate",
    "NumCompaniesWorked": "Number of companies worked for before",
    "Over18": "Always 'Y' (constant)",
    "OverTime": "Works overtime (Yes/No)",
    "PercentSalaryHike": "Percentage salary increase at last hike",
    "PerformanceRating": "Performance rating (1 Low, 2 Good, 3 Excellent, 4 Outstanding)",
    "RelationshipSatisfaction": "Satisfaction with relationships at work (1 Low - 4 Very High)",
    "StandardHours": "Always 80 (constant)",
    "StockOptionLevel": "Stock option level (0 - 3)",
    "TotalWorkingYears": "Total years of work experience",
    "TrainingTimesLastYear": "Number of trainings attended last year",
    "WorkLifeBalance": "Work-life balance (1 Bad, 2 Good, 3 Better, 4 Best)",
    "YearsAtCompany": "Years at the company",
    "YearsInCurrentRole": "Years in the current role",
    "YearsSinceLastPromotion": "Years since the last promotion",
    "YearsWithCurrManager": "Years with the current manager",
}

data_dict = pd.DataFrame({
    "Column": df_raw.columns,
    "Data Type": df_raw.dtypes.astype(str).values,
    "Unique Values": df_raw.nunique().values,
    "Example Value": df_raw.iloc[0].values,
    "Description": [descriptions.get(c, "") for c in df_raw.columns],
})
data_dict.to_csv(os.path.join(OUT_DIR, "data_dictionary.csv"), index=False)
print("\nData dictionary:")
print(data_dict.to_string(index=False))


# TASK 2. DATA CLEANING AND DATA QUALITY ISSUES

heading("TASK 2: DATA CLEANING AND DATA QUALITY")

cleaning_log = []


def log(check, finding, action):
    cleaning_log.append({"Check": check, "Finding": finding, "Action": action})
    print(f"[{check}] {finding} -> {action}")


# 2.1 Missing values
missing = df.isnull().sum()
if missing.sum() == 0:
    log("Missing values", "None found", "No imputation needed")
else:
    for col in missing[missing > 0].index:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())       # numbers -> median
        else:
            df[col] = df[col].fillna(df[col].mode()[0])      # text -> most common value
    log("Missing values", f"{int(missing.sum())} missing cells", "Filled with median / mode")

# 2.2 Duplicates
dup_rows = int(df.duplicated().sum())
dup_ids = int(df["EmployeeNumber"].duplicated().sum())
df = df.drop_duplicates()
log("Duplicates", f"{dup_rows} duplicate rows, {dup_ids} duplicate EmployeeNumbers",
    "Removed duplicate rows" if dup_rows else "Nothing to remove")

# 2.3 Text consistency
text_cols = df.select_dtypes(include=["object", "string"]).columns
for col in text_cols:
    df[col] = df[col].str.strip()
log("Text consistency", "Stripped spaces in text columns", "Unique values checked below")
for col in text_cols:
    print(f"   {col}: {sorted(df[col].unique())}")

# 2.4 Constant columns
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
df = df.drop(columns=constant_cols)
log("Constant columns", f"{constant_cols} have a single value", "Dropped")

# 2.5 Valid ranges and negative values
range_rules = {
    "Education": (1, 5), "EnvironmentSatisfaction": (1, 4), "JobInvolvement": (1, 4),
    "JobSatisfaction": (1, 4), "PerformanceRating": (1, 4), "RelationshipSatisfaction": (1, 4),
    "WorkLifeBalance": (1, 4), "JobLevel": (1, 5), "StockOptionLevel": (0, 3),
}
bad_range = sum(int((~df[c].between(lo, hi)).sum()) for c, (lo, hi) in range_rules.items())
log("Range check", f"{bad_range} values outside valid ranges",
    "Nothing to fix" if bad_range == 0 else "Review these rows")

non_negative = ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears", "YearsAtCompany",
                "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
                "NumCompaniesWorked", "TrainingTimesLastYear"]
neg = int((df[non_negative] < 0).sum().sum())
log("Negative values", f"{neg} negative values", "Nothing to fix" if neg == 0 else "Review these rows")

# 2.6 Logical consistency
logic_rules = {
    "YearsInCurrentRole > YearsAtCompany": df["YearsInCurrentRole"] > df["YearsAtCompany"],
    "YearsWithCurrManager > YearsAtCompany": df["YearsWithCurrManager"] > df["YearsAtCompany"],
    "YearsSinceLastPromotion > YearsAtCompany": df["YearsSinceLastPromotion"] > df["YearsAtCompany"],
    "YearsAtCompany > TotalWorkingYears": df["YearsAtCompany"] > df["TotalWorkingYears"],
    "TotalWorkingYears > Age - 14": df["TotalWorkingYears"] > (df["Age"] - 14),
}
for name, mask in logic_rules.items():
    n = int(mask.sum())
    log("Logic check", f"{n} rows violate: {name}", "Nothing to fix" if n == 0 else "Flag / correct")


# 2.7 Outliers (IQR rule) - flagged but kept, they are real employees
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


for c in ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
          "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
          "TrainingTimesLastYear"]:
    log("Outliers", f"{c}: {iqr_outliers(df[c])} IQR outliers", "Kept (genuine values)")

# 2.8 Target imbalance
share = df["Attrition"].value_counts(normalize=True).mul(100).round(1)
log("Target balance", f"No {share['No']}% / Yes {share['Yes']}%",
    "Imbalanced: compare attrition RATES, not raw counts")

# 2.9 Derived columns (used in the analysis below)
df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)          # 1 = left, 0 = stayed
df["AgeGroup"] = pd.cut(df["Age"], [17, 25, 35, 45, 55, 100],
                        labels=["18-25", "26-35", "36-45", "46-55", "56+"])
df["IncomeBand"] = pd.cut(df["MonthlyIncome"], [0, 3000, 6000, 9000, 12000, 100000],
                          labels=["<3K", "3-6K", "6-9K", "9-12K", "12K+"])
df["TenureBand"] = pd.cut(df["YearsAtCompany"], [-1, 1, 3, 5, 10, 50],
                          labels=["0-1", "2-3", "4-5", "6-10", "10+"])
df["DistanceBand"] = pd.cut(df["DistanceFromHome"], [0, 5, 10, 20, 100],
                            labels=["1-5", "6-10", "11-20", "21+"])
label_maps = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
}
for col, mapping in label_maps.items():
    df[col + "_Label"] = df[col].map(mapping)
log("Feature engineering", "Added AttritionFlag, bands and *_Label columns", "Created")

pd.DataFrame(cleaning_log).to_csv(os.path.join(OUT_DIR, "data_cleaning_report.csv"), index=False)
df.to_csv(os.path.join(OUT_DIR, "HR_Attrition_Cleaned.csv"), index=False)
print(f"\nRaw shape {df_raw.shape} -> Cleaned shape {df.shape}. Files saved in '{OUT_DIR}'.")



# TASK 3. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS

heading("TASK 3: NUMERICAL AND CATEGORICAL COLUMNS")

ordinal_cols = ["Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
                "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
                "StockOptionLevel", "WorkLifeBalance"]           # numeric codes with an order
categorical_cols = list(df.select_dtypes(include=["object", "string"]).columns)
numerical_cols = [c for c in df.select_dtypes(include="number").columns
                  if c not in ordinal_cols + ["EmployeeNumber", "AttritionFlag"]]

print(f"Numerical ({len(numerical_cols)}):", numerical_cols)
print(f"\nOrdinal ({len(ordinal_cols)}):", ordinal_cols)
print(f"\nCategorical ({len(categorical_cols)}):", categorical_cols)
print("\nIdentifier: EmployeeNumber | Target: Attrition (AttritionFlag)")


# TASK 4. EXPLORATORY DATA ANALYSIS (EDA)

heading("TASK 4: EXPLORATORY DATA ANALYSIS")

print("4.1 Summary statistics (numerical columns)")
summary = df[numerical_cols].describe().T
summary["skewness"] = df[numerical_cols].skew()
summary["IQR outliers"] = [iqr_outliers(df[c]) for c in numerical_cols]
print(summary.round(2))
summary.round(2).to_csv(os.path.join(OUT_DIR, "eda_numerical_summary.csv"))

print("\n4.2 Category counts and percentages")
for col in categorical_cols:
    counts = df[col].value_counts()
    print(f"\n--- {col} ---")
    print(pd.DataFrame({"Count": counts, "Percent": (counts / len(df) * 100).round(1)}))

print("\n4.3 Ordinal columns: value counts")
for col in ordinal_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().sort_index())

print("\n4.4 Correlation of each numeric/ordinal column with Attrition (strongest first)")
corr = df[numerical_cols + ordinal_cols + ["AttritionFlag"]].corr()["AttritionFlag"].drop("AttritionFlag")
corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index).round(3)
print(corr_sorted)
corr_sorted.to_csv(os.path.join(OUT_DIR, "eda_correlation_with_attrition.csv"), header=["Correlation"])

print("\n4.5 Strongest correlations between predictors (|r| > 0.5)")
pair_corr = df[numerical_cols + ordinal_cols].corr()
pairs = pair_corr.where(np.triu(np.ones(pair_corr.shape, dtype=bool), k=1)).stack()
print(pairs[pairs.abs() > 0.5].sort_values(ascending=False).round(2))

print("\n4.6 Average of each numeric column: employees who stayed vs left")
compare = df.groupby("Attrition")[numerical_cols].mean().T.round(2)
compare["Difference (Left - Stayed)"] = (compare["Yes"] - compare["No"]).round(2)
print(compare)
compare.to_csv(os.path.join(OUT_DIR, "eda_stayed_vs_left.csv"))



# TASK 5. KEY HR METRICS (KPIs)

heading("TASK 5: KEY HR METRICS (KPIs)")

left_df = df[df["Attrition"] == "Yes"]
stay_df = df[df["Attrition"] == "No"]
total_emp = len(df)
n_left = len(left_df)

kpis = {
    "Total employees": total_emp,
    "Employees who left": n_left,
    "Employees who stayed": total_emp - n_left,
    "Attrition rate (%)": round(n_left / total_emp * 100, 1),
    "Retention rate (%)": round((total_emp - n_left) / total_emp * 100, 1),
    "Average age": round(df["Age"].mean(), 1),
    "Average monthly income": round(df["MonthlyIncome"].mean()),
    "Average monthly income - left": round(left_df["MonthlyIncome"].mean()),
    "Average monthly income - stayed": round(stay_df["MonthlyIncome"].mean()),
    "Average tenure (years)": round(df["YearsAtCompany"].mean(), 1),
    "Average tenure (years) - left": round(left_df["YearsAtCompany"].mean(), 1),
    "Average tenure (years) - stayed": round(stay_df["YearsAtCompany"].mean(), 1),
    "Overtime share overall (%)": round((df["OverTime"] == "Yes").mean() * 100, 1),
    "Overtime share among leavers (%)": round((left_df["OverTime"] == "Yes").mean() * 100, 1),
    "Average years since last promotion": round(df["YearsSinceLastPromotion"].mean(), 1),
    "Average distance from home": round(df["DistanceFromHome"].mean(), 1),
    "Average job satisfaction (1-4)": round(df["JobSatisfaction"].mean(), 2),
    "Average work-life balance (1-4)": round(df["WorkLifeBalance"].mean(), 2),
}
kpi_table = pd.DataFrame(kpis.items(), columns=["KPI", "Value"])
print(kpi_table.to_string(index=False))
kpi_table.to_csv(os.path.join(OUT_DIR, "kpis.csv"), index=False)


# TASK 6. ATTRITION ACROSS EMPLOYEE SEGMENTS

heading("TASK 6: ATTRITION BY EMPLOYEE SEGMENT")


def attrition_by(col, sort_by_rate=True):
    """Headcount, leavers and attrition rate (%) for each group in a column."""
    t = df.groupby(col, observed=True)["AttritionFlag"].agg(
        Employees="count", Left="sum", AttritionRate="mean")
    t["AttritionRate"] = (t["AttritionRate"] * 100).round(1)
    return t.sort_values("AttritionRate", ascending=False) if sort_by_rate else t


segments = {
    "Department": True, "JobRole": True, "Gender": True, "MaritalStatus": True,
    "EducationField": True, "BusinessTravel": True, "JobLevel": False,
    "AgeGroup": False, "TenureBand": False, "Education_Label": True,
}
segment_frames = []
for col, sort_flag in segments.items():
    t = attrition_by(col, sort_flag)
    print(f"\n--- Attrition by {col} ---")
    print(t)
    segment_frames.append(t.reset_index().rename(columns={col: "Group"}).assign(Segment=col))
pd.concat(segment_frames)[["Segment", "Group", "Employees", "Left", "AttritionRate"]].to_csv(
    os.path.join(OUT_DIR, "attrition_by_segment.csv"), index=False)

print("\n--- Attrition rate (%) by Department x JobRole ---")
dept_role = (df.pivot_table(index="Department", columns="JobRole", values="AttritionFlag",
                            aggfunc="mean") * 100).round(1)
print(dept_role.T)

print("\n--- Attrition rate (%) by Gender x MaritalStatus ---")
print((df.pivot_table(index="Gender", columns="MaritalStatus", values="AttritionFlag",
                      aggfunc="mean") * 100).round(1))



# TASK 7. PATTERNS: SALARY, OVERTIME, JOB SATISFACTION, WORK-LIFE BALANCE

heading("TASK 7: SALARY, OVERTIME, JOB SATISFACTION, WORK-LIFE BALANCE")

print("7.1 SALARY")
print("\nAttrition by monthly income band:")
print(attrition_by("IncomeBand", False))
print("\nIncome statistics: left vs stayed")
print(df.groupby("Attrition")["MonthlyIncome"].describe().round(0))
print("\nAverage monthly income by JobLevel and Attrition:")
print(df.pivot_table(index="JobLevel", columns="Attrition", values="MonthlyIncome",
                     aggfunc="mean").round(0))
print("\nAttrition by StockOptionLevel:")
print(attrition_by("StockOptionLevel", False))
print("\nAttrition by PercentSalaryHike group:")
df["HikeBand"] = pd.cut(df["PercentSalaryHike"], [10, 13, 16, 19, 30],
                        labels=["11-13%", "14-16%", "17-19%", "20%+"])
print(attrition_by("HikeBand", False))

print("\n7.2 OVERTIME")
print("\nAttrition by OverTime:")
print(attrition_by("OverTime"))
print("\nAttrition rate (%) by OverTime x MaritalStatus:")
print((df.pivot_table(index="OverTime", columns="MaritalStatus", values="AttritionFlag",
                      aggfunc="mean") * 100).round(1))
print("\nAttrition rate (%) by OverTime x JobLevel:")
print((df.pivot_table(index="OverTime", columns="JobLevel", values="AttritionFlag",
                      aggfunc="mean") * 100).round(1))
print("\nAttrition rate (%) by OverTime x Department:")
print((df.pivot_table(index="OverTime", columns="Department", values="AttritionFlag",
                      aggfunc="mean") * 100).round(1))
print("\nAttrition by BusinessTravel (workload related):")
print(attrition_by("BusinessTravel"))

print("\n7.3 JOB SATISFACTION")
print(attrition_by("JobSatisfaction", False))
print("\nEnvironment satisfaction:")
print(attrition_by("EnvironmentSatisfaction", False))
print("\nJob involvement:")
print(attrition_by("JobInvolvement", False))
print("\nRelationship satisfaction:")
print(attrition_by("RelationshipSatisfaction", False))

print("\n7.4 WORK-LIFE BALANCE")
print(attrition_by("WorkLifeBalance", False))
print("\nAttrition rate (%) by WorkLifeBalance x OverTime:")
print((df.pivot_table(index="WorkLifeBalance", columns="OverTime", values="AttritionFlag",
                      aggfunc="mean") * 100).round(1))
print("\nAttrition by DistanceBand (km from home):")
print(attrition_by("DistanceBand", False))
print("\nAttrition by YearsSinceLastPromotion:")
df["PromoBand"] = pd.cut(df["YearsSinceLastPromotion"], [-1, 0, 2, 5, 20],
                         labels=["0 (this year)", "1-2", "3-5", "6+"])
print(attrition_by("PromoBand", False))

# ---------------- Automatic one-line findings ----------------
heading("KEY FINDINGS (generated from the data)")
ot = attrition_by("OverTime")["AttritionRate"]
inc = attrition_by("IncomeBand", False)["AttritionRate"]
ten = attrition_by("TenureBand", False)["AttritionRate"]
role = attrition_by("JobRole")
js = attrition_by("JobSatisfaction", False)["AttritionRate"]
wlb = attrition_by("WorkLifeBalance", False)["AttritionRate"]
findings = [
    f"Overall attrition: {kpis['Attrition rate (%)']}% ({n_left} of {total_emp}).",
    f"Overtime: {ot['Yes']}% attrition with overtime vs {ot['No']}% without.",
    f"Highest-risk role: {role.index[0]} ({role.iloc[0]['AttritionRate']}%).",
    f"Income: {inc['<3K']}% attrition under 3K/month vs {inc['12K+']}% above 12K.",
    f"Tenure: {ten['0-1']}% attrition in the first year vs {ten['10+']}% after 10+ years.",
    f"Job satisfaction: {js[1]}% attrition at level 1 vs {js[4]}% at level 4.",
    f"Work-life balance: {wlb[1]}% attrition at level 1 (Bad) vs {wlb[3]}% at level 3 (Better).",
]
for i, f in enumerate(findings, 1):
    print(f"{i}. {f}")

print(f"\nAll tables saved in the '{OUT_DIR}' folder.")