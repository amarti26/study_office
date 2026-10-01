# Study Office Risk Support

AAU BDS Session 10 assignment: predict student leaving risk at the end of week 6 and turn the model into decision support for the study office.

## Files

- `app.py` — Streamlit application.
- `portable.py` — portable XGBoost model loader/exporter based on the AAU hotel template.
- `build_model.py` — trains the time-split XGBoost model, exports `model/`, and writes the 2026 top-40 list.
- `model/` — created by `build_model.py`.
- `requirements.txt` — deployment dependencies.

## Run locally

```bash
pip install -r requirements.txt
python build_model.py
streamlit run app.py
```

The app reads the synthetic AAU assignment data directly from the course GitHub repository.

## Decision design

The model is a risk signal. The study office/adviser makes the final decision about contact. The app shows:

1. the ranked 2026 list and top 40;
2. confusion-matrix outcomes for a configurable cut-off on the 2025 validation cohort;
3. precision and recall;
4. performance by international/domestic group;
5. feature contributions for an individual student when the portable model is present.

No student should be contacted automatically from the model output.

## AI/tools disclosure

AI assistance was used to help write and structure code and documentation. The analysis follows the assignment's specified leakage rules, time split, models, cost assumptions, capacity rule and fairness comparison.
