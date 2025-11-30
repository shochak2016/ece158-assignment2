# ece158-assignment2
[Assignment 2 for ECE158 Final](https://docs.google.com/document/d/1w_g-yEtn9lvmXIXQWjQoS51FugM_Mx-ACh0-EZcHWR8/export?format=pdf)
[Amazon Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)

1. Dataset

We will use the Amazon Reviews dataset from McAuley’s website.
We’ll choose one category (e.g., Books, Electronics, Beauty) to keep it manageable.

2. Predictive Task (Our ML Problem)

Goal:
Predict whether an Amazon review is positive (≥4 stars) or negative (≤3 stars) using the review text and (optionally) metadata.

Inputs:
Review text
Optional: summary/title
Optional: userID, itemID

Output:
Binary label: Positive (1) or Negative (0)

Why this task:
Easy to evaluate
Many baseline models from class apply
Plenty of existing research to reference
Fits assignment requirements perfectly

3. Models We Will Implement

Required Baselines
Majority-class baseline (always predict most common class)
Bias model (global mean + user bias + item bias)
TF-IDF + Logistic Regression (our main classical model)

Optional (if time permits / extra credit-level quality)
Linear SVM
DistilBERT fine-tuning on a subset
Model ensembles

4. Evaluation Metrics


We’ll evaluate models using:
Accuracy
Precision + Recall
ROC-AUC
Confusion matrix
ROC curve visualization

We will compare:
Baselines vs TF-IDF model
(Optional) BERT vs TF-IDF

5. Exploratory Data Analysis (EDA)


We will include:
Rating distribution (1–5 stars)
Review length distribution
Word frequency plots
Sample reviews
(Optional) Word cloud
Time-based trends (optional)

We will also:

Clean text (lowercase, remove punctuation)
Create binary labels
Remove missing fields
