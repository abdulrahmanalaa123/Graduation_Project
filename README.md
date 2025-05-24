# 🗄️ Bash-Based DBMS (Lightweight Shell Database Engine)

## Overview

This project is a lightweight **Database Management System (DBMS)** built entirely in **Bash**, simulating core SQL operations without the use of any external database engine. It supports fundamental **DDL** and **DML** features with basic validation and a structured file-based backend.

A key design feature is the use of a **separate metadata file** for each table, which stores column names, data types, and primary key information—just like system catalogs in real DBMSs.

---

## 📚 Supported Features

### 📁 Database Operations
- `CREATE DATABASE <name>`: Creates a new database directory
- `DROP DATABASE <name>`: Deletes the database and all associated tables
- `USE <database>`: Selects the active database
- `SHOW DATABASES`: Lists all available databases

### 📦 Table Operations (DDL)
- `CREATE TABLE <table>`: Creates a new table
- `DROP TABLE <table>`: Deletes the table and its metadata
- `SHOW TABLES`: Lists all tables in the current database
- `DESCRIBE <table>`: Displays schema from the metadata file

> 📄 **Each table has:**
> - `<table>.data`: Contains actual records
> - `<table>.meta`: Stores metadata (column names, types, and primary key)

### 📝 Data Operations (DML)
- `INSERT INTO <table>`: Adds new records with type checking
- `SELECT * FROM <table>`: Displays all data
- `SELECT <columns> FROM <table> WHERE <condition>`: Filtered view
- `UPDATE <table> SET col=val WHERE condition`: Update records
- `DELETE FROM <table> WHERE condition`: Remove records

### ✅ Input Validation
- Enforces primary key constraints using metadata
- Checks column count and data types (e.g., `int`, `str`)
- Clean delimiter-based parsing (default: `|`)

---

## 🧱 System Design

- **Database**: A directory inside `/databases/`
- **Table**: Composed of:
  - `<table>.data`: Actual rows, each line is a record
  - `<table>.meta`: Metadata with schema, types, PKs

**Metadata Example (`students.meta`)**:
```

id\:int\:PK
name\:str
age\:int

```

This enables:
- Centralized schema access
- Enforced column order and type
- Easy `DESCRIBE` and validation logic

---

## 🛠 Project Structure

```

bash-dbms/
├── dbms.sh             # Main CLI interface
├── databases/          # All databases stored here
│   └── school/
│       ├── students.data
│       └── students.meta
└── README.md

````

---

## 🚀 Getting Started

### Prerequisites
- Linux/macOS with Bash
- No external dependencies

### Run the DBMS
```bash
chmod +x dbms.sh
./dbms.sh
````

### Sample Session

```text
CREATE DATABASE school;
USE school;
CREATE TABLE students (id:int:PK, name:str, age:int);
INSERT INTO students VALUES (1, "Alice", 20);
INSERT INTO students VALUES (2, "Bob", 22);
SELECT name FROM students WHERE age > 20;
UPDATE students SET name="Bobby" WHERE id=2;
DELETE FROM students WHERE id=1;
```


## 🧠 Educational Value

This project demonstrates:

* How SQL concepts map to file-based structures
* Custom metadata management like DBMS system catalogs
* Manual enforcement of data integrity rules
* Shell scripting for simulating complex systems

---
## Author
Created by Ahmed Otifi
🔗 GitHub: https://github.com/otifi3
```
