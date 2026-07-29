import sqlite3


connection = sqlite3.connect(
    "data/company.db"
)

cursor = connection.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    FOREIGN KEY(department_id)
    REFERENCES departments(id)
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS salaries (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    salary INTEGER,
    FOREIGN KEY(employee_id)
    REFERENCES employees(id)
)
""")


cursor.executemany(
    """
    INSERT INTO departments VALUES (?,?)
    """,
    [
        (1, "Engineering"),
        (2, "Finance")
    ]
)


cursor.executemany(
    """
    INSERT INTO employees VALUES (?,?,?)
    """,
    [
        (1, "John", 1),
        (2, "Sarah", 1),
        (3, "Mike", 2)
    ]
)


cursor.executemany(
    """
    INSERT INTO salaries VALUES (?,?,?)
    """,
    [
        (1,1,75000),
        (2,2,85000),
        (3,3,65000)
    ]
)


connection.commit()
connection.close()


print("Database created successfully.")