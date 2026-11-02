# Phishing Website Detection

## a. Problem statement
Classify a website as **phishing** or **legitimate** from 30 features extracted from its URL, page source and domain/third-party information (for example IP address in the URL, SSL state, URL length, anchor behaviour, domain age, web traffic and Google index status). Detecting phishing sites automatically helps browsers, mail filters and security teams block credential-theft attacks. Five machine learning classifiers are trained on the same dataset and compared using six evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and the results are explored in an interactive Streamlit app.

## b. Dataset description
- **Source:** Phishing Websites Dataset, UCI Machine Learning Repository (id 327; Mohammad, Thabtah & McCluskey, 2012), https://archive.ics.uci.edu/dataset/327/phishing+websites
- **Task:** binary classification (phishing vs. legitimate)
- **Instances:** 11,055 raw rows (6,157 legitimate, 4,898 phishing); **5,849** after removing exact duplicate rows
- **Features:** 30 integer features, every value in {-1, 0, 1} (1 = legitimate indicator, 0 = suspicious, -1 = phishing indicator). Groups:
  - *Address bar:* having_ip_address, url_length, shortining_service, having_at_symbol, double_slash_redirecting, prefix_suffix, having_sub_domain, sslfinal_state, domain_registration_length, favicon, port, https_token
  - *Abnormal / page content:* request_url, url_of_anchor, links_in_tags, sfh, submitting_to_email, abnormal_url
  - *HTML & JavaScript:* redirect, on_mouseover, rightclick, popupwindow, iframe
  - *Domain:* age_of_domain, dnsrecord, web_traffic, page_rank, google_index, links_pointing_to_page, statistical_report
- **Target:** `Result` (`-1` = phishing, `1` = legitimate). Internally phishing is encoded as the positive class (1).
- **Preprocessing:** dropped missing values and duplicate rows (so identical rows cannot appear in both train and test), standardized features with `StandardScaler` (inside each model pipeline), stratified 80/20 train/test split (`random_state=42`).
- **Split sizes:** train 4,679 rows (2,415 phishing / 2,264 legitimate); test 1,170 rows (604 phishing / 566 legitimate), saved as `test_data.csv`.

## c. GitHub repository link
https://github.com/aishwaryaMokashi-2026aj05223/BITS_ML_Assignment2_PhishingWebsite_Dataset 

Live Streamlit app: <PASTE YOUR STREAMLIT URL HERE>

## d. Models used
All metrics are computed on the held-out test set (`test_data.csv`, 1,170 rows). It is a binary problem, so Precision, Recall and F1 are for the positive class (**Phishing**), and AUC is computed from the predicted probability of phishing.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9179 | 0.9776 | 0.9247 | 0.9156 | 0.9201 | 0.8358 |
| Decision Tree | 0.9239 | 0.9574 | 0.9342 | 0.9172 | 0.9256 | 0.8479 |
| KNN | 0.9179 | 0.9738 | 0.9137 | 0.9288 | 0.9212 | 0.8358 |
| Naive Bayes | 0.6547 | 0.9642 | 0.5992 | 1.0000 | 0.7494 | 0.4141 |
| Support Vector Machine | 0.9444 | 0.9869 | 0.9514 | 0.9404 | 0.9459 | 0.8889 |

Model settings: Logistic Regression (max_iter=3000), Decision Tree (max_depth=12), KNN (k=7), Gaussian Naive Bayes, SVM (RBF kernel, C=10, probability=True). Every model is a pipeline of `StandardScaler` followed by the classifier.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline: 91.8% accuracy and the third-best AUC (0.9776), with balanced precision (0.925) and recall (0.916). It matches KNN on MCC (0.8358) but trails the Decision Tree and SVM, which suggests some non-linear interaction between the features that a linear boundary cannot capture. |
| Decision Tree | Second-best on accuracy (0.9239), F1 (0.9256) and MCC (0.8479), and very interpretable. It has the lowest AUC (0.9574) among the non-Naive-Bayes models, because a tree with limited depth gives coarse probability estimates, so its ranking quality is weaker than its hard predictions. |
| KNN | Accuracy equals Logistic Regression (0.9179) and MCC is the same (0.8358). It has the highest recall among the three simple models (0.9288) but slightly lower precision (0.9137), so it flags a few more legitimate sites as phishing. Prediction is slower than the others because it compares against all stored training rows. |
| Naive Bayes | Clearly the weakest: accuracy 0.6547 and MCC 0.4141. Recall is 1.0 but precision is only 0.5992, so it labels almost every site as phishing and misclassifies most legitimate sites. The AUC is still high (0.9642), meaning it ranks sites well but its decision threshold is badly off. The independence assumption does not hold because many features are strongly correlated, which makes its probabilities over-confident. |
| Support Vector Machine | Best on every metric except recall-by-NB: accuracy 0.9444, AUC 0.9869, precision 0.9514, F1 0.9459 and MCC 0.8889. The RBF kernel captures non-linear interactions between the features, giving the most reliable balance of catching phishing sites and avoiding false alarms. The cost is longer training time than the other models. |

**Overall winner for this dataset: Support Vector Machine** (RBF kernel). It has the highest Accuracy (0.9444), AUC (0.9869), Precision (0.9514), F1 (0.9459) and MCC (0.8889), so it is the best overall and the most balanced on this test set. The Decision Tree is the runner-up.

## Project structure
```
app.py              Streamlit app (loads saved models, no retraining)
train.py            Trains and saves the five models, writes train/test CSVs and results_table.md
utils.py            Target encoding and metric computation
requirements.txt    Dependencies (scikit-learn pinned to 1.6.1)
test_data.csv       Held-out test data for the app (30 features + Result)
train_data.csv      Training split
results_table.md    Metrics table produced by train.py
model/              Saved .joblib pipelines (scaler + classifier) and metrics_table.csv
```

## How to run
```bash
pip install -r requirements.txt
# put the dataset next to train.py (e.g. "Training Dataset.arff" or phishing.csv),
# or pass a path:  python train.py path/to/file.csv
# with no file present, train.py downloads it from UCI via ucimlrepo
python train.py
streamlit run app.py
```
Then upload `test_data.csv` in the app sidebar, pick a model, and view the metrics, the confusion matrix and the classification report. The "All models" tab compares all five models on the uploaded file.

The saved `.joblib` models were trained with scikit-learn 1.6.1, so `requirements.txt` pins that exact version.

