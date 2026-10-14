import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CASE STUDY 2
# SQL + NoSQL DATA INTEGRATION AND MACHINE LEARNING PIPELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SQLITE_DB = DATA_DIR / "student_database.db"
OUTPUT_FILE = OUTPUT_DIR / "model_output.txt"

# MongoDB Atlas credentials are stored outside the source code.
load_dotenv(BASE_DIR / "atlas-credentials.env")
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI was not found in atlas-credentials.env")


# ------------------------------------------------------------
# ORIGINAL SQL DATASET FROM SUBMISSION 1
# ------------------------------------------------------------

students = [
    ('23BCE0046', 'Aarav Sharma', 'BCE', 'CSE - Core', 7, 2027, 8.72, '9001000001'),
    ('23BCE0508', 'Diya Patel', 'BCE', 'CSE - Core', 7, 2027, 9.01, '9001000002'),
    ('23BAI0211', 'Rohan Das', 'BAI', 'CSE - AI & ML', 7, 2027, 8.45, '9001000003'),
    ('23BCE0240', 'Ananya Roy', 'BCE', 'CSE - Core', 7, 2027, 8.88, '9001000004'),
    ('23BCE0038', 'Vikram Singh', 'BCE', 'CSE - Core', 7, 2027, 8.31, '9001000005'),
    ('23BCE0265', 'Meera Iyer', 'BCE', 'CSE - Core', 7, 2027, 9.12, '9001000006'),
    ('23BCE0137', 'Arjun Nair', 'BCE', 'CSE - Core', 7, 2027, 8.67, '9001000007'),
    ('23BCE0223', 'Ishita Gupta', 'BCE', 'CSE - Core', 7, 2027, 8.54, '9001000008'),
    ('23BCE0263', 'Karan Verma', 'BCE', 'CSE - Core', 7, 2027, 7.96, '9001000009'),
    ('23BCE0195', 'Priya Menon', 'BCE', 'CSE - Core', 7, 2027, 8.76, '9001000010'),
    ('23BCE0208', 'Aditya Reddy', 'BCE', 'CSE - Core', 7, 2027, 8.22, '9001000011'),
    ('23BCE0770', 'Sneha Kumar', 'BCE', 'CSE - Core', 7, 2027, 9.03, '9001000012'),
    ('23BCE0085', 'Rahul Jain', 'BCE', 'CSE - Core', 7, 2027, 8.14, '9001000013'),
    ('23BCB0142', 'Kavya Shah', 'BCB', 'CSE - Bioinformatics', 7, 2027, 8.91, '9001000014'),
    ('23BCE0533', 'Nikhil Bose', 'BCE', 'CSE - Core', 7, 2027, 8.38, '9001000015'),
    ('23BCE6767', 'Tanvi Mehta', 'BCE', 'CSE - Core', 7, 2027, 9.24, '9001000016'),
    ('23BCE0768', 'Siddharth Rao', 'BCE', 'CSE - Core', 7, 2027, 7.88, '9001000017'),
    ('23BCE0341', 'Riya Sen', 'BCE', 'CSE - Core', 7, 2027, 8.63, '9001000018'),
    ('23BCI0232', 'Manav Kapoor', 'BCI', 'CSE - Information Security', 7, 2027, 8.49, '9001000019'),
    ('23BCE0131', 'Pooja Sinha', 'BCE', 'CSE - Core', 7, 2027, 8.95, '9001000020'),
    ('23BCI0038', 'Dev Malhotra', 'BCI', 'CSE - Information Security', 7, 2027, 8.17, '9001000021'),
    ('23BCE0049', 'Aditi Chopra', 'BCE', 'CSE - Core', 7, 2027, 8.71, '9001000022'),
    ('23BAI0209', 'Harsh Agarwal', 'BAI', 'CSE - AI & ML', 7, 2027, 8.36, '9001000023'),
    ('23BCE0166', 'Neha Bhat', 'BCE', 'CSE - Core', 7, 2027, 9.08, '9001000024'),
    ('23BKT0058', 'Yash Kulkarni', 'BKT', 'CSE - Blockchain', 7, 2027, 8.02, '9001000025'),
    ('23BCE0567', 'Simran Kaur', 'BCE', 'CSE - Core', 7, 2027, 8.84, '9001000026'),
    ('23BKT0047', 'Kabir Joshi', 'BKT', 'CSE - Blockchain', 7, 2027, 8.59, '9001000027'),
    ('23BCE0187', 'Nandini Mishra', 'BCE', 'CSE - Core', 7, 2027, 9.16, '9001000028'),
    ('23BCE0238', 'Varun Gupta', 'BCE', 'CSE - Core', 7, 2027, 7.74, '9001000029'),
    ('23BCE0243', 'Shreya Das', 'BCE', 'CSE - Core', 7, 2027, 8.69, '9001000030'),
    ('23BCE0176', 'Abhinav Tiwari', 'BCE', 'CSE - Core', 7, 2027, 8.28, '9001000031'),
    ('23BCE1130', 'Muskan Yadav', 'BCE', 'CSE - Core', 7, 2027, 8.93, '9001000032'),
    ('23BCE0057', 'Ayush Srivastava', 'BCE', 'CSE - Core', 7, 2027, 8.51, '9001000033'),
    ('23BCE0084', 'Ira Banerjee', 'BCE', 'CSE - Core', 7, 2027, 8.79, '9001000034'),
    ('23BCE0375', 'Saurabh Pillai', 'BCE', 'CSE - Core', 7, 2027, 8.11, '9001000035'),
    ('23BCE0183', 'Kritika Ghosh', 'BCE', 'CSE - Core', 7, 2027, 9.31, '9001000036'),
    ('23BCE0769', 'Aman Thakur', 'BCE', 'CSE - Core', 7, 2027, 8.04, '9001000037'),
    ('23BCE0207', 'Sanya Arora', 'BCE', 'CSE - Core', 7, 2027, 8.65, '9001000038'),
    ('23BCE0308', 'Dhruv Bansal', 'BCE', 'CSE - Core', 7, 2027, 8.27, '9001000039'),
    ('23BKT0169', 'Manya Dutta', 'BKT', 'CSE - Blockchain', 7, 2027, 9.02, '9001000040'),
    ('23BCE0277', 'Ritvik Saxena', 'BCE', 'CSE - Core', 7, 2027, 8.46, '9001000041'),
    ('23BCE0342', 'Nisha Prasad', 'BCE', 'CSE - Core', 7, 2027, 8.73, '9001000042'),
    ('23BCE0124', 'Ankit Mandal', 'BCE', 'CSE - Core', 7, 2027, 7.92, '9001000043'),
    ('23BCE2124', 'Palak Jha', 'BCE', 'CSE - Core', 7, 2027, 8.86, '9001000044'),
    ('23BCE0045', 'Khushi Sharma', 'BCE', 'CSE - Core', 7, 2027, 8.18, '9001000045'),
    ('23BCE0396', 'Anushka Sarkar', 'BCE', 'CSE - Core', 7, 2027, 9.19, '9001000046'),
    ('23BCE0078', 'Rajat Mitra', 'BCE', 'CSE - Core', 7, 2027, 8.57, '9001000047'),
    ('23BCE0028', 'Lavanya Krishnan', 'BCE', 'CSE - Core', 7, 2027, 8.97, '9001000048'),
    ('23BCE1046', 'Sahil Khan', 'BCE', 'CSE - Core', 7, 2027, 8.33, '9001000049'),
    ('23BCE0060', 'Isha Ahuja', 'BCE', 'CSE - Core', 7, 2027, 8.78, '9001000050'),
    ('23BCE0865', 'Vivek Pandey', 'BCE', 'CSE - Core', 7, 2027, 8.42, '9001000051'),
    ('23BCE0036', 'Ayesha Ali', 'BCE', 'CSE - Core', 7, 2027, 8.89, '9001000052'),
    ('23BCE0479', 'Om Deshmukh', 'BCE', 'CSE - Core', 7, 2027, 8.06, '9001000053'),
    ('23BCE2304', 'Radhika Iyer', 'BCE', 'CSE - Core', 7, 2027, 9.05, '9001000054'),
    ('23BCE0116', 'Varsha Nair', 'BCE', 'CSE - Core', 7, 2027, 8.61, '9001000055'),
    ('23BCE0858', 'Kunal Bhatt', 'BCE', 'CSE - Core', 7, 2027, 8.74, '9001000056'),
    ('23BKT0034', 'Anirudh Roy', 'BKT', 'CSE - Blockchain', 7, 2027, 8.24, '9001000057'),
    ('23BCE0822', 'Sakshi Pawar', 'BCE', 'CSE - Core', 7, 2027, 9.11, '9001000058'),
    ('23BCE0789', 'Pranav Seth', 'BCE', 'CSE - Core', 7, 2027, 8.35, '9001000059'),
    ('23BCE0992', 'Jhanvi Rana', 'BCE', 'CSE - Core', 7, 2027, 8.81, '9001000060'),
    ('23BCE0334', 'Rishabh Soni', 'BCE', 'CSE - Core', 7, 2027, 8.13, '9001000061'),
    ('23BAI0315', 'Aditi Rao', 'BAI', 'CSE - AI & ML', 7, 2027, 8.67, '9001000062'),
    ('23BAI0421', 'Rohit Mehra', 'BAI', 'CSE - AI & ML', 7, 2027, 8.94, '9001000063'),
    ('23BCI0175', 'Sahil Verma', 'BCI', 'CSE - Information Security', 7, 2027, 8.58, '9001000064'),
    ('23BCI0284', 'Mihir Joshi', 'BCI', 'CSE - Information Security', 7, 2027, 9.06, '9001000065'),
    ('23BCB0193', 'Nikita Kapoor', 'BCB', 'CSE - Bioinformatics', 7, 2027, 8.47, '9001000066'),
    ('23BCB0271', 'Rhea Chatterjee', 'BCB', 'CSE - Bioinformatics', 7, 2027, 8.83, '9001000067'),
    ('23BKT0218', 'Ishaan Malhotra', 'BKT', 'CSE - Blockchain', 7, 2027, 8.71, '9001000068'),
    ('23BKT0296', 'Pavithra Krishnan', 'BKT', 'CSE - Blockchain', 7, 2027, 8.55, '9001000069'),
    ('23BAI0357', 'Adarsh Kumar', 'BAI', 'CSE - AI & ML', 7, 2027, 9.18, '9001000070')
]

courses = [
    ("BCSE401L", "Design and Analysis of Algorithms", "CSE - Core", 4),
    ("BCSE402L", "Database Management Systems", "CSE - Core", 4),
    ("BCSE403L", "Computer Networks", "CSE - Core", 4),
    ("BCSE404L", "Operating Systems", "CSE - Core", 4),
    ("BCSE405L", "Machine Learning", "CSE - AI & ML", 4),
    ("BCSE406L", "Deep Learning", "CSE - AI & ML", 3),
    ("BCSE407L", "Cyber Security", "CSE - Information Security", 4),
    ("BCSE408L", "Network Security", "CSE - Information Security", 3),
    ("BCSE409L", "Computational Biology", "CSE - Bioinformatics", 4),
    ("BCSE410L", "Blockchain Technology", "CSE - Blockchain", 4),
]


def create_original_sql_database():
    """Recreate the original normalized SQLite database."""

    connection = sqlite3.connect(SQLITE_DB)
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.executescript("""
        DROP TABLE IF EXISTS grades;
        DROP TABLE IF EXISTS enrollments;
        DROP TABLE IF EXISTS courses;
        DROP TABLE IF EXISTS students;

        CREATE TABLE students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            degree_code TEXT NOT NULL,
            specialization TEXT NOT NULL,
            semester INTEGER NOT NULL,
            batch INTEGER NOT NULL,
            cgpa REAL NOT NULL,
            phone TEXT NOT NULL UNIQUE
        );

        CREATE TABLE courses (
            course_id TEXT PRIMARY KEY,
            course_name TEXT NOT NULL,
            specialization TEXT NOT NULL,
            credits INTEGER NOT NULL
        );

        CREATE TABLE enrollments (
            enrollment_id INTEGER PRIMARY KEY,
            student_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            enrollment_year INTEGER NOT NULL,
            UNIQUE(student_id, course_id),
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (course_id) REFERENCES courses(course_id)
        );

        CREATE TABLE grades (
            grade_id INTEGER PRIMARY KEY,
            enrollment_id INTEGER NOT NULL UNIQUE,
            marks REAL NOT NULL,
            grade TEXT NOT NULL,
            result TEXT NOT NULL,
            FOREIGN KEY (enrollment_id) REFERENCES enrollments(enrollment_id)
        );
    """)

    cursor.executemany("""
        INSERT INTO students
        (student_id, student_name, degree_code, specialization,
         semester, batch, cgpa, phone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, students)

    cursor.executemany("""
        INSERT INTO courses
        (course_id, course_name, specialization, credits)
        VALUES (?, ?, ?, ?)
    """, courses)

    # Same enrollment rule as the original SQL project:
    # a student is enrolled in courses matching the specialization.
    cursor.execute("""
        INSERT INTO enrollments
        (enrollment_id, student_id, course_id, enrollment_year)
        SELECT
            ROW_NUMBER() OVER (ORDER BY s.student_id, c.course_id),
            s.student_id,
            c.course_id,
            2026
        FROM students s
        JOIN courses c
            ON s.specialization = c.specialization
    """)

    # Same deterministic grade-generation rule as the original project.
    cursor.execute("""
        INSERT INTO grades
        (grade_id, enrollment_id, marks, grade, result)
        SELECT
            e.enrollment_id,
            e.enrollment_id,
            ROUND(
                CASE
                    WHEN (e.enrollment_id % 10) = 0 THEN 62
                    WHEN (e.enrollment_id % 10) = 1 THEN 78
                    WHEN (e.enrollment_id % 10) = 2 THEN 85
                    WHEN (e.enrollment_id % 10) = 3 THEN 91
                    WHEN (e.enrollment_id % 10) = 4 THEN 73
                    WHEN (e.enrollment_id % 10) = 5 THEN 88
                    WHEN (e.enrollment_id % 10) = 6 THEN 67
                    WHEN (e.enrollment_id % 10) = 7 THEN 95
                    WHEN (e.enrollment_id % 10) = 8 THEN 81
                    ELSE 70
                END + ((e.enrollment_id % 7) - 3), 2
            ),
            CASE
                WHEN (
                    CASE
                        WHEN (e.enrollment_id % 10) = 0 THEN 62
                        WHEN (e.enrollment_id % 10) = 1 THEN 78
                        WHEN (e.enrollment_id % 10) = 2 THEN 85
                        WHEN (e.enrollment_id % 10) = 3 THEN 91
                        WHEN (e.enrollment_id % 10) = 4 THEN 73
                        WHEN (e.enrollment_id % 10) = 5 THEN 88
                        WHEN (e.enrollment_id % 10) = 6 THEN 67
                        WHEN (e.enrollment_id % 10) = 7 THEN 95
                        WHEN (e.enrollment_id % 10) = 8 THEN 81
                        ELSE 70
                    END + ((e.enrollment_id % 7) - 3)
                ) >= 90 THEN 'A+'
                WHEN (
                    CASE
                        WHEN (e.enrollment_id % 10) = 0 THEN 62
                        WHEN (e.enrollment_id % 10) = 1 THEN 78
                        WHEN (e.enrollment_id % 10) = 2 THEN 85
                        WHEN (e.enrollment_id % 10) = 3 THEN 91
                        WHEN (e.enrollment_id % 10) = 4 THEN 73
                        WHEN (e.enrollment_id % 10) = 5 THEN 88
                        WHEN (e.enrollment_id % 10) = 6 THEN 67
                        WHEN (e.enrollment_id % 10) = 7 THEN 95
                        WHEN (e.enrollment_id % 10) = 8 THEN 81
                        ELSE 70
                    END + ((e.enrollment_id % 7) - 3)
                ) >= 80 THEN 'A'
                WHEN (
                    CASE
                        WHEN (e.enrollment_id % 10) = 0 THEN 62
                        WHEN (e.enrollment_id % 10) = 1 THEN 78
                        WHEN (e.enrollment_id % 10) = 2 THEN 85
                        WHEN (e.enrollment_id % 10) = 3 THEN 91
                        WHEN (e.enrollment_id % 10) = 4 THEN 73
                        WHEN (e.enrollment_id % 10) = 5 THEN 88
                        WHEN (e.enrollment_id % 10) = 6 THEN 67
                        WHEN (e.enrollment_id % 10) = 7 THEN 95
                        WHEN (e.enrollment_id % 10) = 8 THEN 81
                        ELSE 70
                    END + ((e.enrollment_id % 7) - 3)
                ) >= 70 THEN 'B+'
                WHEN (
                    CASE
                        WHEN (e.enrollment_id % 10) = 0 THEN 62
                        WHEN (e.enrollment_id % 10) = 1 THEN 78
                        WHEN (e.enrollment_id % 10) = 2 THEN 85
                        WHEN (e.enrollment_id % 10) = 3 THEN 91
                        WHEN (e.enrollment_id % 10) = 4 THEN 73
                        WHEN (e.enrollment_id % 10) = 5 THEN 88
                        WHEN (e.enrollment_id % 10) = 6 THEN 67
                        WHEN (e.enrollment_id % 10) = 7 THEN 95
                        WHEN (e.enrollment_id % 10) = 8 THEN 81
                        ELSE 70
                    END + ((e.enrollment_id % 7) - 3)
                ) >= 60 THEN 'B'
                ELSE 'C'
            END,
            CASE
                WHEN (
                    CASE
                        WHEN (e.enrollment_id % 10) = 0 THEN 62
                        WHEN (e.enrollment_id % 10) = 1 THEN 78
                        WHEN (e.enrollment_id % 10) = 2 THEN 85
                        WHEN (e.enrollment_id % 10) = 3 THEN 91
                        WHEN (e.enrollment_id % 10) = 4 THEN 73
                        WHEN (e.enrollment_id % 10) = 5 THEN 88
                        WHEN (e.enrollment_id % 10) = 6 THEN 67
                        WHEN (e.enrollment_id % 10) = 7 THEN 95
                        WHEN (e.enrollment_id % 10) = 8 THEN 81
                        ELSE 70
                    END + ((e.enrollment_id % 7) - 3)
                ) >= 50 THEN 'PASS'
                ELSE 'FAIL'
            END
        FROM enrollments e
    """)

    connection.commit()
    return connection


def extract_sql_features(connection):
    """Aggregate the relational database into student-level features."""

    query = """
        SELECT
            s.student_id,
            s.student_name,
            s.specialization,
            s.semester,
            s.batch,
            s.cgpa,
            COUNT(DISTINCT e.course_id) AS courses_enrolled,
            ROUND(AVG(g.marks), 2) AS average_marks
        FROM students s
        JOIN enrollments e ON s.student_id = e.student_id
        JOIN grades g ON e.enrollment_id = g.enrollment_id
        GROUP BY
            s.student_id,
            s.student_name,
            s.specialization,
            s.semester,
            s.batch,
            s.cgpa
        ORDER BY s.student_id
    """

    return pd.read_sql_query(query, connection)


def create_mongodb_interaction_logs(client):
    """Create semi-structured nested interaction documents in MongoDB."""

    database = client["fods_case_study_2"]
    collection = database["interaction_logs"]

    # Re-running the program should not duplicate old records.
    collection.delete_many({})

    documents = []

    for position, student in enumerate(students, start=1):
        document = {
            "student_id": student[0],
            "activity": {
                "login_count": 5 + ((position * 3) % 16),
                "course_views": 10 + ((position * 5) % 31),
                "assignment_submissions": 3 + ((position * 2) % 8),
                "quiz_attempts": 2 + ((position * 4) % 9),
                "forum_posts": (position * 3) % 7,
                "avg_session_minutes": 20 + ((position * 7) % 41)
            },
            "metadata": {
                "platform": "Learning Portal",
                "semester": student[4]
            }
        }

        # Deliberate missing values demonstrate the required
        # missing-value handling during preprocessing.
        if position in (7, 21, 42):
            document["activity"]["forum_posts"] = None

        if position in (13, 35, 56):
            document["activity"]["avg_session_minutes"] = None

        documents.append(document)

    collection.insert_many(documents)
    return collection


def extract_nosql_features(collection):
    """Extract and flatten the nested MongoDB documents."""

    documents = list(collection.find({}, {"_id": 0}))

    records = []

    for document in documents:
        activity = document.get("activity", {})

        records.append({
            "student_id": document.get("student_id"),
            "login_count": activity.get("login_count"),
            "course_views": activity.get("course_views"),
            "assignment_submissions": activity.get("assignment_submissions"),
            "quiz_attempts": activity.get("quiz_attempts"),
            "forum_posts": activity.get("forum_posts"),
            "avg_session_minutes": activity.get("avg_session_minutes")
        })

    return pd.DataFrame(records)


def train_and_evaluate(dataset):
    """Preprocess data, train Logistic Regression and evaluate it."""

    # Define the target from the SQL academic outcome.
    # average_marks is NOT included in X, preventing direct target leakage.
    dataset["performance_category"] = dataset["average_marks"].apply(
        lambda value: "High Performance"
        if value >= 75
        else "Low Performance"
    )

    feature_columns = [
        "specialization",
        "semester",
        "batch",
        "cgpa",
        "courses_enrolled",
        "login_count",
        "course_views",
        "assignment_submissions",
        "quiz_attempts",
        "forum_posts",
        "avg_session_minutes"
    ]

    X = dataset[feature_columns]
    y = dataset["performance_category"]

    categorical_features = ["specialization"]
    numerical_features = [
        column for column in feature_columns
        if column not in categorical_features
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return (
        y_test,
        predictions,
        accuracy_score(y_test, predictions),
        precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


def main():
    output_lines = []

    def log(message=""):
        print(message)
        output_lines.append(str(message))

    log("=" * 70)
    log("CASE STUDY 2 - SQL + NoSQL MACHINE LEARNING PIPELINE")
    log("=" * 70)

    # --------------------------------------------------------
    # 1. SQL DATA EXTRACTION
    # --------------------------------------------------------

    connection = create_original_sql_database()
    sql_df = extract_sql_features(connection)

    cursor = connection.cursor()

    table_counts = {}
    for table in ("students", "courses", "enrollments", "grades"):
        table_counts[table] = cursor.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

    log("\n1. SQL DATA SOURCE")
    log("-" * 70)
    log(f"Students    : {table_counts['students']}")
    log(f"Courses     : {table_counts['courses']}")
    log(f"Enrollments : {table_counts['enrollments']}")
    log(f"Grades      : {table_counts['grades']}")

    # --------------------------------------------------------
    # 2. NOSQL DATA EXTRACTION
    # --------------------------------------------------------

    log("\n2. NOSQL DATA SOURCE")
    log("-" * 70)

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=10000
    )

    client.admin.command("ping")
    log("MongoDB Atlas connection: SUCCESS")

    collection = create_mongodb_interaction_logs(client)
    nosql_df = extract_nosql_features(collection)

    log(f"Interaction documents: {len(nosql_df)}")

    # --------------------------------------------------------
    # 3. MERGE DATASETS
    # --------------------------------------------------------

    merged_df = pd.merge(
        sql_df,
        nosql_df,
        on="student_id",
        how="inner"
    )

    log("\n3. DATA INTEGRATION")
    log("-" * 70)
    log("Common key: student_id")
    log(f"Merged dataset shape: {merged_df.shape}")

    # --------------------------------------------------------
    # 4. PREPROCESSING
    # --------------------------------------------------------

    log("\n4. PREPROCESSING")
    log("-" * 70)

    missing_values = merged_df.isnull().sum()

    log("Missing values before imputation:")

    if missing_values.sum() > 0:
        log(
            missing_values[missing_values > 0].to_string()
        )
    else:
        log("None")

    # --------------------------------------------------------
    # 5. MACHINE LEARNING
    # --------------------------------------------------------

    log("\n5. MODEL IMPLEMENTATION")
    log("-" * 70)
    log("Model: Logistic Regression")
    log("Train/Test split: 70% / 30%")

    y_test, predictions, accuracy, precision, recall = (
        train_and_evaluate(merged_df)
    )

    # --------------------------------------------------------
    # 6. EVALUATION
    # --------------------------------------------------------

    log("\n6. MODEL EVALUATION")
    log("-" * 70)
    log(f"Accuracy : {accuracy:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall   : {recall:.4f}")

    # --------------------------------------------------------
    # 7. FINAL MODEL OUTCOME
    # --------------------------------------------------------

    log("\n7. FINAL MODEL OUTCOME")
    log("-" * 70)

    prediction_table = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    log(prediction_table.to_string(index=False))

    # --------------------------------------------------------
    # 8. INTERPRETATION
    # --------------------------------------------------------

    log("\n8. INTERPRETATION")
    log("-" * 70)

    log(
        f"The Logistic Regression model achieved {accuracy:.2%} "
        f"accuracy, {precision:.2%} weighted precision and "
        f"{recall:.2%} weighted recall on the 30% test set. "
        "The model uses integrated academic and interaction "
        "features after missing-value imputation, categorical "
        "encoding and numerical scaling."
    )

    log("\n" + "=" * 70)
    log("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    log("=" * 70)

    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8"
    )

    connection.close()
    client.close()


if __name__ == "__main__":
    main()
