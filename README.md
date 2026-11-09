# Wine Quality Prediction System

A Python/Streamlit classroom prototype using a Random Forest classifier.

## Important dataset note
The included `wine_quality.csv` is a **50-row preview** transcribed from the dataset viewer, not the complete 1,599-row UCI red-wine dataset. The included `wine_model.pkl` was trained on this preview only so the project can be opened and tested. Do not report its evaluation as the full-dataset lab result.

For the final laboratory submission, download the complete red-wine CSV from the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), replace `wine_quality.csv` (save it comma-separated with headers), then run `python train_model.py` again to create the final `wine_model.pkl`.

## Prediction rule
- quality >= 6: GOOD QUALITY
- quality < 6: NEEDS IMPROVEMENT
- `quality` is excluded from model input features.

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The Streamlit app runs locally at `http://localhost:8501`. Upload a CSV with the 11 physicochemical input columns.

## Limitations
This is a classroom prototype and not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
