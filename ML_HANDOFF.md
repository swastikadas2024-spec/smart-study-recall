# ML Handoff – Smart Study Recall

## 1. What has been completed

The initial ML pipeline for Smart Study Recall has been developed.

Current work includes:

- Duolingo Learning Traces dataset exploration
- Data cleaning and chronological train/validation/test splitting
- HLR-based retention prediction prototype
- Forgetting-risk calculation
- Revision-priority logic
- Logistic Regression recall model experiment
- Random Forest recall regression experiment
- Initial calibration experiment
- Python prediction pipeline

## 2. Dataset

Dataset used:

Duolingo Learning Traces / 13M learning traces.

The raw dataset is intentionally NOT included in GitHub because it is large.

Place the dataset at:

ml/data/raw/settles.acl16.learning_traces.13m.csv.gz

The dataset contains fields including:

- user_id
- lexeme_id
- timestamp
- delta
- history_seen
- history_correct
- session_seen
- session_correct
- p_recall

## 3. Current HLR model

The current saved HLR parameters are:

ml/models/hlr_parameters.json

The current HLR implementation uses:

- log_history_seen
- historical_accuracy
- elapsed time in days

The prediction pipeline is implemented in:

ml/src/prediction.py

## 4. Important: HLR is NOT final

The current HLR model is a prototype/baseline.

Its predicted half-life values can become unrealistically large, causing retention predictions to remain very close to 1 for many inputs.

Therefore:

DO NOT treat the current half-life or recommended revision days as scientifically validated.

The model needs further calibration/design work before being used as the final scheduling model.

## 5. Other models tested

### Logistic Regression

File:

ml/src/recall_model.py

This model predicts a binary recall outcome derived from p_recall.

It was evaluated against the continuous p_recall value, so this experiment should be treated as exploratory rather than the final model.

### Random Forest Regression

Files:

ml/src/recall_regression.py
ml/src/train_recall_regression.py

This model directly predicts p_recall.

The model was evaluated using MAE and RMSE.

IMPORTANT:

The current JSON file for this experiment stores model metadata, not the fitted Random Forest itself.

Therefore the Random Forest cannot simply be loaded for prediction from that JSON. It would need to be retrained or properly serialized.

## 6. Calibration

Calibration was attempted using isotonic regression.

Files:

ml/src/calibrate_recall.py
ml/src/train_calibration.py

The calibration pipeline is NOT currently confirmed as working.

If continuing this approach, first fix and run the calibration pipeline and compare:

- raw HLR
- calibrated HLR
- Random Forest regression

using the same chronological test set.

## 7. Current revision policy

File:

ml/src/revision_policy.py

Current thresholds:

Retention < 0.60
? Review now

Retention 0.60–0.80
? Review soon

Retention >= 0.80
? Review later

These thresholds are currently heuristic and should be validated before being treated as final.

## 8. Important dataset consideration

p_recall represents recall/performance information from the learning interaction.

Some session-level fields can contain information closely related to the current recall outcome.

Therefore, avoid using current-session variables such as session_correct/session_seen for a rigorous pre-recall prediction model because they can introduce target leakage.

Historical features are preferred for predicting future retention.

## 9. What to do next

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

## 10. Core architecture

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

## 11. Important warning

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

