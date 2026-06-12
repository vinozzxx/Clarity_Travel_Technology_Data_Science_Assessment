# Clarity Travel Technology Solutions
## Data Science Assessment Dashboard

An end-to-end data science assessment built on the Clarity TTS airline booking dataset (2,000 records).

---

## Project Structure

```
├── app.py                    # Streamlit dashboard (main application)
├── rag_engine.py             # RAG engine (ChromaDB + Groq Llama-3)
├── 01_EDA.ipynb              # Part 1 — Exploratory Data Analysis
├── 02_Model.ipynb            # Part 2 — Cancellation Prediction Model
├── 03_NLP_Bonus.ipynb        # Part 3 — NLP Complaint Classification
├── data/
│   ├── clarity_bookings_dataset.csv    # Raw dataset (2,000 bookings)
│   └── cleaned_clarity_bookings.csv    # Cleaned dataset
├── model/
│   ├── logistic_regression_model.pkl   # Trained LR model
│   └── xgb_classifier_model.pkl        # Trained XGBoost model
└── requirements.txt          # Python dependencies
```

---

## Assessment Deliverables

### Part 1 — Business Insights (EDA)
- Revenue analysis by airline, route, cabin class and booking channel
- Cancellation pattern analysis (by channel, cabin, lead time, trip type)
- Monthly booking trends and seasonal demand patterns
- New vs Repeat customer behaviour comparison

### Part 2 — Cancellation Prediction
- Binary classification: predict cancelled/refunded bookings
- Models: Logistic Regression (baseline) + XGBoost (production)
- Pipeline: Scikit-learn ColumnTransformer (imputation + encoding + scaling)
- Evaluation: ROC-AUC, Precision, Recall, F1, Confusion Matrix

### Part 3 — Complaint Intelligence (NLP Bonus)
- Rule-based NLP classifier: 9 operational complaint categories
- Severity scoring (High / Medium / Low) with team routing
- AI executive summary generation via Groq Llama-3

### AI Data Assistant
- Natural language Q&A over the entire dataset
- Architecture: Pre-computed metrics → Groq Llama-3 → Structured business answer
- No vector DB required — full 2,000-row metrics as context

---

## Setup

```bash
# 1. Clone repository
git clone https://github.com/vinozzxx/Clarity_Travel_Technology_Data_Science_Assessment.git
cd Clarity_Travel_Technology_Data_Science_Assessment

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
# Create a .env file (NOT committed — see .gitignore):
echo GROQ_API_KEY=your_groq_api_key_here > .env

# 5. Run the dashboard
.venv\Scripts\python.exe -m streamlit run app.py
```

> **Get a free Groq API key:** https://console.groq.com

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-Learn | ML pipeline |
| XGBoost | Gradient boosted classifier |
| Streamlit | Interactive dashboard |
| Matplotlib / Seaborn | Visualisations |
| Groq API (Llama-3) | LLM generation |
| ChromaDB | Vector store (RAG engine) |
| Sentence Transformers | Text embeddings |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Project Overview | KPI summary and feature cards |
| 📊 Revenue Analysis | Airline, route, cabin, NDC vs GDS revenue |
| ❌ Cancellation Analysis | Cancellation patterns and risk factors |
| 📅 Booking Trends | Monthly volume and channel mix |
| 👤 Customer Behaviour | New vs Repeat customer comparison |
| 🤖 Booking Risk Predictor | Live cancellation probability scorer |
| 📈 Model Performance | ROC curves, confusion matrix, metrics table |
| 🔬 Feature Importance | Top predictive features with business context |
| 💬 Live Complaint Classifier | Real-time complaint categorisation |
| 📋 Batch Complaint Analysis | Dataset-wide complaint heatmaps |
| 🧠 AI Complaint Summariser | Executive summary via Groq Llama-3 |
| 🔍 AI Data Assistant | Natural language Q&A over the dataset |
| ℹ️ About Project | Full methodology and tech stack |

---

*Clarity Travel Technology Solutions — Data Science Assessment Submission, June 2026*
