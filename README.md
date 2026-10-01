# Semester at the Study Office

Completed AAU BDS Session 10 assignment: predict student leaving risk at the end of week 6 and turn the model into decision support for the study office.

## Repository contents

- `M1_10_assignment_study_office_COMPLETED_EXECUTED.ipynb` — completed and executed assignment notebook.
- `app.py` — Streamlit decision-support app.
- `portable.py` — portable XGBoost model loader/exporter.
- `model/preprocess.json` and `model/booster.json` — exported model.
- `contact_list_2026_top40.csv` — 2026 top-40 adviser-review list.
- `data/history_week6.csv` and `data/new_week6.csv` — assignment data.
- `requirements.txt` — Python dependencies.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app uses the exported portable model when `model/` is present. The model is a risk signal; the study office/adviser makes the final contact decision.

## Analytical design

- Leakage is controlled so only variables available at the end of week 6 enter the model.
- Training uses 2023–2024; validation uses 2025.
- Logistic regression and XGBoost are compared by validation AUC.
- XGBoost is used for the portable app.
- Thresholds are evaluated with confusion-matrix metrics and the assignment's stated cost assumptions.
- A fixed top-40 capacity rule and international/domestic group comparison are included.
- The app supports human review and does not automatically contact students.

## AI/tools disclosure

AI assistance was used to help structure code, documentation, and implementation. The numerical results in the notebook are calculated from the supplied assignment data using the specified workflow and assumptions.
