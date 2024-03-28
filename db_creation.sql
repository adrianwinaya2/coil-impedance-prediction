CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predict_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    pid_lv REAL,
    lid_lv REAL,
    tid_lv REAL,
    pod_lv REAL,
    lod_lv REAL,
    tod_lv REAL,
    pid_hv REAL,
    lid_hv REAL,
    tid_hv REAL,
    pod_hv REAL,
    lod_hv REAL,
    tod_hv REAL,
    impedance REAL
)
-- sqlite3 your_database.db < your_sql_file.sql