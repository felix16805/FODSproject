# FODS Project — Interactive Streamlit Dashboard

## Overview

This application analyses uploaded Excel workbooks and CSV files
and can also read the existing SQLite student database.

## Existing project files

The existing project structure is preserved:

- data/student_database.db
- output/model_output.txt
- .env.example
- atlas-credentials.env
- main.py

Additional files:

- ml_pipeline.py
- requirements.txt
- README.md
- README_streamlit.md
- .gitignore

## Installation

Open a terminal in the FODSproject directory and run:

    python -m pip install -r requirements.txt

## Run the dashboard

    python -m streamlit run main.py

Open the local URL printed by Streamlit, usually:

    http://localhost:8501

## Upload a dataset

1. Select Upload Excel / CSV in the sidebar.
2. Upload an .xlsx, .xls, or .csv file.
3. Select a worksheet if the workbook contains multiple sheets.
4. Explore the KPI cards, charts, filters and data-quality report.
5. Download filtered records as CSV if required.

## Existing SQL database

Select Use existing SQLite project in the sidebar.

The dashboard reads data/student_database.db in read-only mode.
It expects students, courses, enrollments and grades tables.

## Notes

- Uploaded files are analysed in memory.
- The app does not automatically save uploaded files.
- A time-series chart requires usable date/time data.
- The dashboard does not access Atlas credentials.
- Never commit atlas-credentials.env to version control.
