# ML Handoff – Smart Study Recall

## 1\. What has been completed

The initial ML pipeline for Smart Study Recall has been developed.

Current work includes:

* Duolingo Learning Traces dataset exploration
* Data cleaning and chronological train/validation/test splitting
* HLR-based retention prediction prototype
* Forgetting-risk calculation
* Revision-priority logic
* Logistic Regression recall model experiment
* Random Forest recall regression experiment
* Initial calibration experiment
* Python prediction pipeline

## 2\. Dataset

Dataset used:

Duolingo Learning Traces / 13M learning traces.

The raw dataset is intentionally NOT included in GitHub because it is large.

Place the dataset at:

ml/data/raw/settles.acl16.learning\_traces.13m.csv.gz

The dataset contains fields including:

* user\_id
* lexeme\_id
* timestamp
* delta
* history\_seen
* history\_correct
* session\_seen
* session\_correct
* p\_recall

## 3\. Current HLR model

The current saved HLR parameters are:

ml/models/hlr\_parameters.json

The current HLR implementation uses:

* log\_history\_seen
* historical\_accuracy
* elapsed time in days

The prediction pipeline is implemented in:

ml/src/prediction.py

## 4\. Important: HLR is NOT final

The current HLR model is a prototype/baseline.

Its predicted half-life values can become unrealistically large, causing retention predictions to remain very close to 1 for many inputs.

Therefore:

DO NOT treat the current half-life or recommended revision days as scientifically validated.

The model needs further calibration/design work before being used as the final scheduling model.

## 5\. Other models tested

### Logistic Regression

File:

ml/src/recall\_model.py

This model predicts a binary recall outcome derived from p\_recall.

It was evaluated against the continuous p\_recall value, so this experiment should be treated as exploratory rather than the final model.

### Random Forest Regression

Files:

ml/src/recall\_regression.py
ml/src/train\_recall\_regression.py

This model directly predicts p\_recall.

The model was evaluated using MAE and RMSE.

IMPORTANT:

The current JSON file for this experiment stores model metadata, not the fitted Random Forest itself.

Therefore the Random Forest cannot simply be loaded for prediction from that JSON. It would need to be retrained or properly serialized.

## 6\. Calibration

Calibration was attempted using isotonic regression.

Files:

ml/src/calibrate\_recall.py
ml/src/train\_calibration.py

The calibration pipeline is NOT currently confirmed as working.

If continuing this approach, first fix and run the calibration pipeline and compare:

* raw HLR
* calibrated HLR
* Random Forest regression

using the same chronological test set.

## 7\. Current revision policy

File:

ml/src/revision\_policy.py

Current thresholds:

Retention < 0.60
? Review now

Retention 0.60–0.80
? Review soon

Retention >= 0.80
? Review later

These thresholds are currently heuristic and should be validated before being treated as final.

## 8\. Important dataset consideration

p\_recall represents recall/performance information from the learning interaction.

Some session-level fields can contain information closely related to the current recall outcome.

Therefore, avoid using current-session variables such as session\_correct/session\_seen for a rigorous pre-recall prediction model because they can introduce target leakage.

Historical features are preferred for predicting future retention.

## 9\. What to do next

Recommended order:

1. Fix/complete the calibration pipeline.
2. Compare HLR, calibrated HLR and Random Forest regression.
3. Use a strict chronological evaluation.
4. Check calibration and prediction distributions, not only MAE/RMSE.
5. Decide the final retention model.
6. Build a clean prediction function for the backend.
7. Create the FastAPI ML endpoint.
8. Connect the endpoint to the application backend.
9. Implement the final revision scheduling logic.
10. Integrate recall-quiz results so future predictions can update based on student performance.

## 10\. Core architecture

Student interaction
?
Historical learning data
?
Retention prediction model
?
Predicted retention / forgetting risk
?
Revision policy
?
Recommended revision time
?
Recall quiz
?
New performance data
?
Updated prediction

## 11\. Important warning

Do not claim that the current model accurately predicts when a student will forget something.

The current implementation is an ML prototype and requires further validation/calibration.

The final goal is:

Personalized retention prediction
+
Recall performance feedback
+
Prerequisite/dependency awareness
+
Adaptive revision scheduling



\## GRU v2 Experiment Results



\*\*Dataset:\*\* First 200,000 rows of the Duolingo Learning Traces dataset, after excluding 2 rows with tied learner-item timestamps.



\*\*Sequence construction:\*\* Events grouped by learner and lexeme, with sequence length 10.



\*\*Split:\*\* Chronological 70% training, 15% validation, and 15% test.



| Model | Test MAE | Test RMSE |

|---|---:|---:|

| GRU v2 | 0.173775 | 0.273756 |

| RF-1 | 0.172481 | 0.271126 |

| Mean baseline | 0.177073 | 0.277290 |



\*\*Conclusion:\*\* GRU v2 outperformed the mean baseline but did not outperform RF-1 on either test metric. RF-1 remains the preferred model among these experiments. Results are preliminary and based on a limited dataset sample; further validation is required before claiming generalization or superiority.



\*\*Saved artifacts:\*\*

\- `ml/models/recall\_gru\_v2.pt`

\- `ml/models/recall\_gru\_v2\_metadata.json`



\*\*Target clarification:\*\* The model predicts an observed session-level recall rate. This should not be presented as a validated estimate of long-term retention or forgetting probability without further evaluation.

