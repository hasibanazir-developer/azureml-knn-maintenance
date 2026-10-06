# Predictive Maintenance with KNN on Azure Machine Learning
Student: Hasiba Nazir, 0045

## Problem
Predict machine failure from sensor readings (AI4I 2020 dataset, UCI, CC BY 4.0). Only 3.4% of machines fail, so models are judged on recall, F1 and AUC, not accuracy.

## Approaches
1. Notebook (scikit-learn + MLflow): notebooks/01_knn_notebook.ipynb
2. Automated ML (KNN only): automl/automl_results.md
3. Designer (Execute Python Script): designer/knn_designer_script.py

## Results
| | Notebook | Automated ML | Designer |
|---|---|---|---|
| Best K | 1 | 75 | 1 |
| Weights | uniform | distance | uniform |
| Scaler | StandardScaler | SparseNormalizer | StandardScaler |
| Test recall (failure) | 0.353 | 0.555 (macro) | see Designer job output |
| Test F1 (failure) | 0.397 | 0.590 (macro) | see Designer job output |
| AUC | 0.669 | 0.923 | see Designer job output |
| Code written | Most | Little (SDK) | Small script |
| Deployable to managed endpoint | Yes | Yes | No (classic components) |

## Answers to report questions
- Most useful sensors: torque and tool wear. Failed machines show higher torque, lower rpm and higher tool wear.
- Scaling: accuracy stayed about 0.97 with and without scaling, but recall rose from 0.206 to 0.294, because accuracy is dominated by the 96.6% healthy machines.
- Threshold: lowering the threshold catches more failures at the cost of more false alarms. For maintenance a lower threshold (e.g. 0.3) is better, because a missed failure costs more than an extra inspection.
- Designer vs notebook: metrics can differ because Split Data creates a different train/test split from scikit-learn.
- Best approach: AutoML gave the best AUC, while the notebook gives full control and was deployed. For a real factory I would use AutoML to search and the notebook for control and deployment.
- Imbalance: 3.4% failures is hard for KNN. More failure data, a lower threshold, or class weighting in another algorithm would help.

## Deployment
Managed online endpoint knn-maint-hasiba0045c, deployment "blue", VM Standard_DS2_v2, tested from the SDK, the studio Test tab and a REST call (HTTP 200, prediction [0, 1]). Deleted after testing to stop charges.
- The first no-code MLflow deployment crashed (container exit code 3) because the serving environment used different package versions from training. Fixed with a custom environment pinned to the training versions and a small scoring script (deployment/score.py).
- Sample request: deployment/sample-request.json
- Test script: deployment/test_endpoint.py

## What I learned
KNN needs scaled features because it is distance based. Accuracy is misleading on imbalanced data, so recall, F1 and AUC matter more. AutoML failed on sparse categorical data with KNN, which I fixed with ordinal encoding. The metric you optimise (F1 vs AUC) changes the chosen K a lot. Training and deployment environments must use the same package versions. Azure resources cost money, so idle shutdown and deleting resources is essential.

## Screenshots
See the screenshots/ folder.
