# DB to store training results and logging.
#    Builds a SQLite DB with a table to store model training results.
#    Tables include 'runs' and 'predictions'
#    Table columns include:
#        id: Primary key
#        model_name: Name of the model architecture
#        train_accuracy: Training accuracy after all epochs
#        val_accuracy: Validation accuracy after all epochs
#        test_accuracy: Test accuracy after all epochs
#        train_loss: Training loss after all epochs
#        val_loss: Validation loss after all epochs
#        test_loss: Test loss after all epochs
#        train_precision: Training precision after all epochs
#        train_recall: Training recall after all epochs
#        f1: Metric used to evaluate classification model score after all epochs
#        timestamp: Timestamp of the training session
#        model_path: File path to the saved trained model
#        confidence: Confidence score of the model predictions
import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict

# Path to DB
DB_PATH = './output/results.db'

# Define and return DB connection
def get_db_connection(db_path: str = DB_PATH):
    return sqlite3.connect(db_path)

# Build tables
def create_tables(db_path: str = DB_PATH):
    # If tables do not exist, create them
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # Table: runs to store training results
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
        
            train_accuracy REAL,
            train_loss REAL,
            train_precision REAL,
            train_recall REAL,
            train_f1 REAL,

            val_accuracy REAL,
            val_loss REAL,
            val_precision REAL,
            val_recall REAL,
            val_f1 REAL,

            test_accuracy REAL,
            test_loss REAL,
            test_precision REAL,
            test_recall REAL,
            test_f1 REAL,

            timestamp TEXT NOT NULL,
            model_path TEXT
        );
        '''
    )
    # Table: predictions to store individual image prediction results
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            image_path TEXT,
            true_label TEXT,
            predicted_label TEXT,
            confidence REAL,
            correct INTEGER,
            timestamp TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );
        '''
    )
    conn.commit()
    conn.close()

# Insert training run results into the runs table
def insert_run(
    model_name: str,
    train_accuracy: float, train_loss: float, train_precision: float,
    train_recall: float, train_f1: float,
    val_accuracy: float, val_loss: float, val_precision: float,
    val_recall: float, val_f1: float,
    test_accuracy: float, test_loss: Optional[float], test_precision: float,
    test_recall: float, test_f1: float,
    timestamp: str, model_path: str,
    db_path: str = DB_PATH
) -> int:
    # Add row containing metrics of completed training run
    conn = get_db_connection(db_path)
    cur = conn.cursor()


    cur.execute(
        '''
        INSERT INTO runs (
            model_name,
            train_accuracy, train_loss, train_precision, train_recall, train_f1,
            val_accuracy, val_loss, val_precision, val_recall, val_f1,
            test_accuracy, test_loss, test_precision, test_recall, test_f1,
            timestamp,
            model_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''',
        (
        model_name,
        train_accuracy, train_loss, train_precision, train_recall, train_f1,
        val_accuracy, val_loss, val_precision, val_recall, val_f1,
        test_accuracy, test_loss, test_precision, test_recall, test_f1,
        timestamp, model_path
        )
    )

    run_id = cur.lastrowid # Get the ID of the inserted row
    conn.commit()
    conn.close()
    return run_id

# Insert prediction results into the predictions table
def insert_prediction(run_id: int, image_path: str, true_label: str,
                      predicted_label: str, confidence: float, correct: int, db_path=DB_PATH):
    # Add row containing prediction result
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        '''
        INSERT INTO predictions (
            run_id,
            image_path,
            true_label,
            predicted_label,
            confidence,
            correct,
            timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        ''',
        (
            run_id,
            image_path,
            true_label,
            predicted_label,
            confidence,
            int(correct),
            datetime.now().isoformat()
        ),
    )

    conn.commit()
    conn.close()

# Query all training runs from the runs table
def query_all_runs(db_path: str = DB_PATH):
    """
    Query all training runs from the runs table.
    Returns a list of dictionaries, one per row.
    """
    conn = get_db_connection(db_path)
    conn.row_factory = sqlite3.Row  # to get dict-like access
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs ORDER BY id ASC;")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]