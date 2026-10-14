from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SQLITE_DB = DATA_DIR / "student_database.db"

st.set_page_config(
    page_title="FODS | Student Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Styling ----------------------------

st.markdown(
    """
    <style>
      .block-container {
          max-width: 1500px;
          padding-top: 1.35rem;
          padding-bottom: 2.2rem;
      }

      [data-testid="stMetric"] {
          background: linear-gradient(135deg, #101c2c, #172b3d);
          border: 1px solid #294154;
          padding: 16px 18px;
          border-radius: 15px;
      }

      [data-testid="stMetricLabel"] p {
          color: #b9c8d8 !important;
      }

      [data-testid="stMetricValue"] {
          color: #f6fbff !important;
      }

      .hero {
          padding: 1.2rem 1.35rem;
          border-radius: 18px;
          background: linear-gradient(110deg, #102638, #164e63);
          border: 1px solid #286276;
          margin-bottom: 1rem;
      }

      .hero h1 {
          margin: 0;
          color: #f3fbff;
          font-size: 2rem;
      }

      .hero p {
          margin: .35rem 0 0 0;
          color: #c8e5ef;
      }

      .section-note {
          color: #7d8da1;
          font-size: .88rem;
      }

      div[data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: 15px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------- Data loading ----------------------------

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and remove completely empty rows/columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(axis=0, how="all").dropna(axis=1, how="all")


def read_uploaded_file(uploaded_file, sheet_name=None) -> pd.DataFrame:
    """Read an uploaded Excel or CSV file."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")

        return normalise_columns(df)

    if name.endswith((".xlsx", ".xls")):
        excel = pd.ExcelFile(io.BytesIO(raw))
        chosen_sheet = (
            sheet_name
            if sheet_name in excel.sheet_names
            else excel.sheet_names[0]
        )
        return normalise_columns(
            pd.read_excel(excel, sheet_name=chosen_sheet)
        )

    raise ValueError(
        "Unsupported file. Upload an Excel workbook or CSV file."
    )


def load_local_sql_data() -> Optional[pd.DataFrame]:
    """Read the existing SQLite database without modifying it."""
    if not SQLITE_DB.exists():
        return None

    try:
        uri = f"file:{SQLITE_DB.as_posix()}?mode=ro"

        with sqlite3.connect(uri, uri=True) as con:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                con,
            )["name"].tolist()

            required = {
                "students",
                "courses",
                "enrollments",
                "grades",
            }

            if not required.issubset(set(tables)):
                return None

            query = """
                SELECT
                    s.student_id,
                    s.student_name,
                    s.degree_code,
                    s.specialization,
                    s.semester,
                    s.batch,
                    s.cgpa,
                    c.course_id,
                    c.course_name,
                    c.credits,
                    e.enrollment_id,
                    e.enrollment_year,
                    g.marks,
                    g.grade,
                    g.result
                FROM students s
                JOIN enrollments e
                    ON e.student_id = s.student_id
                JOIN courses c
                    ON c.course_id = e.course_id
                JOIN grades g
                    ON g.enrollment_id = e.enrollment_id
            """

            return normalise_columns(
                pd.read_sql_query(query, con)
            )

    except (sqlite3.Error, pd.errors.DatabaseError):
        return None


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """Find a column using common alternative names."""
    by_lower = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in by_lower:
            return by_lower[key]

    return None


def to_numeric_if_useful(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking text columns while preserving identifiers."""
    result = df.copy()

    for col in result.columns:
        name = str(col).lower()

        if any(
            token in name
            for token in [
                "id", "phone", "mobile", "zip",
                "postal", "code",
            ]
        ):
            continue

        if result[col].dtype == "object":
            converted = pd.to_numeric(
                result[col].astype(str).str.replace(
                    ",", "", regex=False
                ),
                errors="coerce",
            )

            nonempty = result[col].notna().sum()

            if (
                nonempty
                and converted.notna().sum() / nonempty >= 0.85
            ):
                result[col] = converted

    return result


def infer_date_column(df: pd.DataFrame) -> Optional[str]:
    """Identify a usable date/time field if one exists."""
    preferred = find_column(
        df,
        [
            "date",
            "datetime",
            "timestamp",
            "created_at",
            "created_date",
            "enrollment_date",
            "exam_date",
            "month",
            "year",
        ],
    )

    if preferred:
        parsed = pd.to_datetime(
            df[preferred], errors="coerce"
        )

        if parsed.notna().sum() >= max(
            2, int(len(df) * 0.5)
        ):
            df[preferred] = parsed
            return preferred

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

        if any(
            token in str(col).lower()
            for token in ["date", "time", "timestamp"]
        ):
            parsed = pd.to_datetime(
                df[col], errors="coerce"
            )

            if parsed.notna().sum() >= max(
                2, int(len(df) * 0.5)
            ):
                df[col] = parsed
                return col

    return None


def safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns with more than one distinct value."""
    return [
        col
        for col in df.select_dtypes(include="number").columns
        if df[col].nunique(dropna=True) > 1
    ]


# ---------------------------- Header ----------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📊 Student & Dataset Analytics</h1>
        <p>
            Upload an Excel workbook or CSV, explore the data,
            and generate interactive insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------- Sidebar and source selection ----------------------------

with st.sidebar:
    st.header("Data source")

    source = st.radio(
        "Choose a dataset",
        [
            "Upload Excel / CSV",
            "Use existing SQLite project",
        ],
        index=0,
        help=(
            "The SQLite option reads "
            "data/student_database.db in read-only mode."
        ),
    )

    uploaded = None
    selected_sheet = None

    if source == "Upload Excel / CSV":
        uploaded = st.file_uploader(
            "Upload your dataset",
            type=["xlsx", "xls", "csv"],
            help="Excel workbooks with multiple sheets are supported.",
        )

        if uploaded and uploaded.name.lower().endswith(
            (".xlsx", ".xls")
        ):
            try:
                sheet_names = pd.ExcelFile(
                    io.BytesIO(uploaded.getvalue())
                ).sheet_names

                selected_sheet = st.selectbox(
                    "Worksheet",
                    sheet_names,
                )

            except Exception as exc:
                st.error(
                    f"Could not read workbook sheets: {exc}"
                )

    st.divider()

    st.caption(
        "Your existing data/, output/, and credentials "
        "files are not modified by this dashboard."
    )


# ---------------------------- Load selected data ----------------------------

df: Optional[pd.DataFrame] = None
source_label: str = ""

try:
    if source == "Upload Excel / CSV":
        if uploaded is None:
            st.info(
                "Upload an Excel (.xlsx/.xls) or CSV file "
                "using the left sidebar to start the analysis."
            )

            st.markdown("### What the dashboard can do")

            a, b, c = st.columns(3)

            a.markdown(
                "**Explore**\n\n"
                "Preview rows, column types, missing values, "
                "and summary statistics."
            )

            b.markdown(
                "**Visualise**\n\n"
                "Automatically create numeric, category, "
                "and date-based charts when supported."
            )

            c.markdown(
                "**Export**\n\n"
                "Download the filtered dataset after "
                "applying the dashboard filters."
            )

            st.stop()

        df = read_uploaded_file(
            uploaded,
            selected_sheet,
        )

        source_label = (
            f"Uploaded: {uploaded.name}"
            + (
                f" · Sheet: {selected_sheet}"
                if selected_sheet
                else ""
            )
        )

    else:
        df = load_local_sql_data()

        if df is None:
            st.error(
                "Could not read the expected SQL tables from "
                "data/student_database.db. Check that the "
                "database contains students, courses, "
                "enrollments, and grades."
            )
            st.stop()

        source_label = (
            "Existing SQLite database · student_database.db"
        )

except Exception as exc:
    st.error(f"Unable to load the selected data: {exc}")
    st.stop()


if df is None or df.empty:
    st.warning(
        "The selected file contains no usable rows. "
        "Please upload a non-empty spreadsheet or CSV."
    )
    st.stop()


df = to_numeric_if_useful(normalise_columns(df))
date_col = infer_date_column(df)

st.caption(
    f"**Source:** {source_label}  ·  "
    f"**Rows:** {len(df):,}  ·  "
    f"**Columns:** {len(df.columns):,}"
)


# ---------------------------- Filters ----------------------------

st.sidebar.header("Dashboard filters")

filtered = df.copy()

category_cols = [
    col
    for col in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    if 1 < df[col].nunique(dropna=True) <= 100
]

filter_cols = category_cols[:3]

with st.sidebar:
    for col in filter_cols:
        values = sorted(
            df[col].dropna().astype(str).unique().tolist()
        )

        selected = st.multiselect(
            col,
            values,
            default=values,
            key=f"filter_{col}",
        )

        if selected:
            filtered = filtered[
                filtered[col].astype(str).isin(selected)
            ]
        else:
            filtered = filtered.iloc[0:0]

    if date_col:
        dates = pd.to_datetime(
            df[date_col], errors="coerce"
        ).dropna()

        if not dates.empty:
            min_date = dates.min().date()
            max_date = dates.max().date()

            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

            if (
                isinstance(date_range, tuple)
                and len(date_range) == 2
            ):
                start_date, end_date = date_range

                parsed_dates = pd.to_datetime(
                    filtered[date_col], errors="coerce"
                )

                filtered = filtered[
                    parsed_dates.dt.date.between(
                        start_date, end_date
                    )
                ]


if filtered.empty:
    st.warning(
        "No rows match the current filters. "
        "Select one or more values in the sidebar."
    )
    st.stop()


# ---------------------------- Detect useful fields ----------------------------

numeric_cols = safe_numeric_columns(filtered)

student_col = find_column(
    filtered,
    [
        "student_id",
        "studentid",
        "roll_no",
        "roll_number",
        "registration_id",
    ],
)

marks_col = find_column(
    filtered,
    [
        "marks",
        "mark",
        "score",
        "percentage",
        "average_marks",
        "total_marks",
        "grade_point",
    ],
)

cgpa_col = find_column(
    filtered,
    ["cgpa", "gpa", "cumulative_gpa"],
)

course_col = find_column(
    filtered,
    ["course_name", "course", "subject", "module"],
)

specialization_col = find_column(
    filtered,
    [
        "specialization",
        "department",
        "category",
        "program",
        "branch",
    ],
)


# ---------------------------- KPI cards ----------------------------

k1, k2, k3, k4 = st.columns(4)

k1.metric("Rows analysed", f"{len(filtered):,}")
k2.metric("Columns", f"{len(filtered.columns):,}")

if student_col:
    k3.metric(
        "Unique students",
        f"{filtered[student_col].nunique(dropna=True):,}",
    )
elif category_cols:
    k3.metric(
        "Categories",
        f"{filtered[category_cols[0]].nunique(dropna=True):,}",
    )
else:
    k3.metric("Numeric fields", f"{len(numeric_cols):,}")

if (
    marks_col
    and pd.api.types.is_numeric_dtype(filtered[marks_col])
):
    k4.metric(
        f"Average {marks_col}",
        f"{filtered[marks_col].mean():,.2f}",
    )

elif (
    cgpa_col
    and pd.api.types.is_numeric_dtype(filtered[cgpa_col])
):
    k4.metric(
        "Average CGPA",
        f"{filtered[cgpa_col].mean():,.2f}",
    )

elif numeric_cols:
    k4.metric(
        f"Average {numeric_cols[0]}",
        f"{filtered[numeric_cols[0]].mean():,.2f}",
    )

else:
    k4.metric(
        "Missing cells",
        f"{int(filtered.isna().sum().sum()):,}",
    )


# ---------------------------- Interactive charts ----------------------------

st.markdown("## Interactive visual analysis")

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("Performance / numeric breakdown")

        if (
            course_col
            and marks_col
            and pd.api.types.is_numeric_dtype(
                filtered[marks_col]
            )
        ):
            chart_data = (
                filtered.groupby(
                    course_col, dropna=False
                )[marks_col]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean")
            )

            chart_data.columns = [
                course_col,
                "Average",
                "Records",
            ]

            fig = px.bar(
                chart_data,
                x="Average",
                y=course_col,
                orientation="h",
                color="Average",
                text=chart_data["Average"].round(2),
                hover_data={"Records": True},
                color_continuous_scale="Teal",
                labels={
                    course_col: "Course / category",
                    "Average": f"Average {marks_col}",
                },
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                height=410,
                margin=dict(
                    l=8, r=28, t=20, b=10
                ),
                coloraxis_showscale=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        elif numeric_cols:
            chosen_num = st.selectbox(
                "Numeric measure",
                numeric_cols,
                key="numeric_breakdown",
            )

            if category_cols:
                chosen_cat = st.selectbox(
                    "Group by",
                    category_cols,
                    key="category_breakdown",
                )

                chart_data = (
                    filtered.groupby(
                        chosen_cat, dropna=False
                    )[chosen_num]
                    .mean()
                    .reset_index()
                    .sort_values(chosen_num)
                )

                fig = px.bar(
                    chart_data,
                    x=chosen_num,
                    y=chosen_cat,
                    orientation="h",
                    color=chosen_num,
                    color_continuous_scale="Teal",
                )

                fig.update_layout(
                    height=410,
                    margin=dict(
                        l=8, r=24, t=20, b=10
                    ),
                    coloraxis_showscale=False,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:
                fig = px.histogram(
                    filtered,
                    x=chosen_num,
                    nbins=20,
                    title=f"Distribution of {chosen_num}",
                )

                fig.update_layout(
                    height=410,
                    margin=dict(
                        l=8, r=20, t=35, b=10
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        else:
            st.info(
                "No numeric columns were detected "
                "for a numeric chart."
            )


with right:
    with st.container(border=True):
        st.subheader("Category breakdown")

        cat_for_chart = (
            specialization_col
            or (
                category_cols[0]
                if category_cols
                else None
            )
        )

        if cat_for_chart:
            count_data = (
                filtered[cat_for_chart]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .rename_axis(cat_for_chart)
                .reset_index(name="Records")
            )

            count_data = count_data.head(15)

            fig = px.bar(
                count_data.sort_values("Records"),
                x="Records",
                y=cat_for_chart,
                orientation="h",
                color="Records",
                color_continuous_scale="Blues",
            )

            fig.update_layout(
                height=410,
                margin=dict(
                    l=8, r=20, t=20, b=10
                ),
                coloraxis_showscale=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        elif numeric_cols:
            fig = px.box(
                filtered,
                y=numeric_cols[0],
                points="outliers",
                title=f"Spread of {numeric_cols[0]}",
            )

            fig.update_layout(
                height=410,
                margin=dict(
                    l=8, r=20, t=35, b=10
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.info(
                "No suitable categorical column "
                "was detected."
            )


# ---------------------------- Time-series trend ----------------------------

st.markdown("## Trend analysis")

if date_col and numeric_cols:
    trend_measure = (
        marks_col
        if marks_col and marks_col in numeric_cols
        else st.selectbox(
            "Measure for trend",
            numeric_cols,
            key="trend_measure",
        )
    )

    trend_df = filtered.copy()

    trend_df[date_col] = pd.to_datetime(
        trend_df[date_col], errors="coerce"
    )

    trend_df = trend_df.dropna(
        subset=[date_col, trend_measure]
    )

    if not trend_df.empty:
        trend_df["Period"] = (
            trend_df[date_col]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        trend = (
            trend_df.groupby(
                "Period", as_index=False
            )[trend_measure]
            .mean()
        )

        fig = px.line(
            trend,
            x="Period",
            y=trend_measure,
            markers=True,
            title=f"{trend_measure} over time",
        )

        fig.update_layout(
            height=340,
            margin=dict(
                l=8, r=20, t=40, b=10
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:
        st.info(
            "A date column exists, but there are not "
            "enough valid date/measure pairs to show a trend."
        )

else:
    st.info(
        "No reliable date/time column was detected. "
        "A true time-series chart cannot be generated "
        "from this dataset. Upload data with a date, "
        "timestamp, month, or year field to enable trend "
        "analysis. No dates are fabricated."
    )


# ---------------------------- Numeric correlation heatmap ----------------------------

if len(numeric_cols) >= 2:
    with st.expander(
        "Explore relationships between numeric fields"
    ):
        corr = filtered[numeric_cols].corr(
            numeric_only=True
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric correlation heatmap",
        )

        fig.update_layout(
            height=max(
                350,
                min(650, 45 * len(numeric_cols)),
            ),
            margin=dict(
                l=8, r=8, t=45, b=8
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ---------------------------- Data quality report ----------------------------

with st.expander("Data quality report"):
    quality = pd.DataFrame(
        {
            "Column": filtered.columns,
            "Data type": [
                str(filtered[col].dtype)
                for col in filtered.columns
            ],
            "Missing values": [
                int(filtered[col].isna().sum())
                for col in filtered.columns
            ],
            "Missing %": [
                round(
                    float(
                        filtered[col].isna().mean() * 100
                    ),
                    2,
                )
                for col in filtered.columns
            ],
            "Unique values": [
                int(filtered[col].nunique(dropna=True))
                for col in filtered.columns
            ],
        }
    )

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Total missing cells in filtered data: "
        f"{int(filtered.isna().sum().sum()):,}"
    )


# ---------------------------- Records and export ----------------------------

st.markdown("## Records")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    height=350,
)

csv_bytes = filtered.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Download filtered data as CSV",
    data=csv_bytes,
    file_name="filtered_analysis.csv",
    mime="text/csv",
)


st.caption(
    "Results describe the uploaded or local dataset only. "
    "Correlation does not prove causation, and time trends "
    "are shown only when actual date/time data is available."
)
