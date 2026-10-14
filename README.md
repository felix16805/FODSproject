# 📊 FODS Student Analytics & Machine Learning Pipeline

An end-to-end data science and analytics platform featuring an **interactive Streamlit analytical dashboard** and an **integrated SQL + NoSQL machine learning pipeline**.

Designed for analyzing academic datasets, visualizing student performance metrics, and modeling student outcomes using combined relational and behavioral data.

---

## 🌟 Key Features

### 1. Interactive Streamlit Dashboard (`main.py`)
- **Multi-Source Ingestion**:
  - Upload external **Excel spreadsheets** (`.xlsx`, `.xls`) with automated worksheet detection.
  - Upload **CSV files** with multi-encoding fallback (`utf-8`, `latin-1`).
  - Read existing normalized relational databases ([`data/student_database.db`](data/student_database.db)) in **read-only mode**.
- **Dynamic KPI Metrics**: Immediate overview of total records, analyzed attributes, unique student counts, and average marks/CGPA.
- **Visual Analytics**: Interactive Plotly horizontal bar charts, numeric distributions, categorical breakdowns, and correlation heatmaps.
- **Automated Data Quality Profiling**: Instant audit of missing values, fill rates, column data types, and unique cardinality.
- **Multi-Dimensional Filtering**: Categorical slicers and dynamic date range filters with real-time UI updates.
- **Filtered Export**: One-click CSV export of filtered views with UTF-8 BOM encoding.

### 2. Integrated SQL + NoSQL ML Pipeline (`ml_pipeline.py`)
- **Relational Data Modeling (SQLite)**: 4 normalized tables (`students`, `courses`, `enrollments`, `grades`) with primary/foreign keys.
- **Semi-Structured Document Ingestion (MongoDB Atlas)**: Captures nested portal activity logs (logins, session minutes, forum participation, assignment submissions).
- **Data Integration & Preprocessing**:
  - Inner join on `student_id`.
  - Missing-value imputation via `SimpleImputer` (median for numeric, mode for categorical).
  - Feature normalization via `StandardScaler` and categorical encoding via `OneHotEncoder`.
- **Classification Modeling**: Predicts academic performance category (`High Performance` vs. `Low Performance`) via balanced `LogisticRegression` without target leakage.

---

## 🏗️ Architecture & How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                       Data Sources                          │
├──────────────────────────────┬──────────────────────────────┤
│ Relational (SQLite / Excel)  │  NoSQL (MongoDB Atlas / JSON)│
│  - Academic records          │  - Learning portal activity  │
│  - Courses & grades          │  - Forum posts & logins      │
└──────────────┬───────────────┴──────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐┌──────────────────────────────┐
│  Streamlit App (main.py)     ││ ML Pipeline (ml_pipeline.py) │
│  - In-memory data loading    ││ - Feature extraction         │
│  - Read-only SQLite query    ││ - Common key merge           │
│  - Dynamic KPI computation   ││ - Scikit-learn Pipeline      │
│  - Plotly interactive charts ││ - Evaluation & output log    │
│  - Data Quality reporting    ││                              │
└──────────────┬───────────────┘└──────────────┬───────────────┘
               │                               │
               ▼                               ▼
    Interactive Web Dashboard             Model Performance
     (http://localhost:8501)            (output/model_output.txt)
```

---

## 🔒 Security & Credentials Management

This repository is configured to **strictly prevent leaking API keys, MongoDB connection strings, or environment secrets**:

- `atlas-credentials.env` and `.env` are explicitly ignored in [`.gitignore`](.gitignore).
- A sample template [`.env.example`](.env.example) is provided for configuring local secrets.
- The Streamlit web dashboard operates **completely in memory** and only reads local SQLite in read-only mode (`?mode=ro`), requiring zero database write permissions or cloud credentials.
- When deploying to **Streamlit Community Cloud** or **Docker**, credentials can be injected safely via environment variables or `secrets.toml`.

---

## 📁 Repository Structure

```
FODSproject/
├── .env.example              # Template for environment variables (safe for git)
├── .gitignore                # Protects credentials, venvs, and Python cache
├── README.md                 # Primary documentation and system overview
├── README_streamlit.md       # Dedicated Streamlit usage guide
├── requirements.txt          # Python library dependencies
├── main.py                   # Streamlit dashboard application
├── ml_pipeline.py            # Preserved SQL + NoSQL ML training pipeline
├── data/
│   └── student_database.db   # Local SQLite academic database
└── output/
    └── model_output.txt      # Model evaluation results and execution logs
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/felix16805/FODSproject.git
cd FODSproject
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Launch the Streamlit Dashboard
```bash
python -m streamlit run main.py
```
Open your browser to:
```
http://localhost:8501
```

#### Dashboard Usage Guide:
1. **Choose a Data Source** from the sidebar:
   - **Upload Excel / CSV**: Drag and drop any spreadsheet (`.xlsx`, `.xls`, `.csv`). Select your desired worksheet if the workbook has multiple sheets.
   - **Use existing SQLite project**: Automatically loads the pre-configured [`data/student_database.db`](data/student_database.db).
2. **Apply Filters**: Filter records by categorical attributes or date ranges in the sidebar.
3. **Explore Visualizations**:
   - Compare performance across categories.
   - Inspect numeric distributions and correlation heatmaps.
   - Audit missing data in the **Data quality report** expander.
4. **Export Results**: Click **Download filtered data as CSV** to save filtered records.

---

### Option B: Execute the SQL + NoSQL ML Pipeline
1. Create your `atlas-credentials.env` file from the template:
   ```bash
   cp .env.example atlas-credentials.env
   ```
2. Populate your `MONGODB_URI` connection string inside `atlas-credentials.env`.
3. Run the pipeline:
   ```bash
   python ml_pipeline.py
   ```
4. Check [`output/model_output.txt`](output/model_output.txt) for generated performance logs, accuracy scores, and classification outcomes.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard framework |
| `plotly` | Dynamic interactive charting and data visualization |
| `pandas` | High-performance tabular data manipulation |
| `openpyxl` & `xlrd` | Excel spreadsheet parsing (`.xlsx` and `.xls`) |
| `scikit-learn` | Preprocessing, imputation, model training & metrics |
| `pymongo` | MongoDB Atlas driver for NoSQL logs |
| `python-dotenv` | Secure local environment variable management |

---

## 👤 Author

Developed by **[@felix16805](https://github.com/felix16805)**.
