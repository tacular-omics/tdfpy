from tdfpy import slice_d_folder

slice_d_folder("/home/patrick-garrett/Data/Arabela/seminal_plasma/raw/DIA/boar/20220217_Boar-1_S3-A7_1_8655.d", "tests/data/example_dia.d", frame_start=1, frame_end=500)

# open this sqllite db and pritn size of each table in mb: tests/data/example_dia.d/analysis.tdf
import sqlite3

def print_table_sizes(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT SUM(pgsize) FROM dbstat WHERE name='{table_name}';")
        size_bytes = cursor.fetchone()[0]
        size_mb = size_bytes / (1024 * 1024)
        print(f"Table: {table_name}, Size: {size_mb:.2f} MB")

    conn.close()

print_table_sizes("tests/data/example_dia.d/analysis.tdf")