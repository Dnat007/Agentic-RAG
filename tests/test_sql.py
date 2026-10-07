import sqlite3

import pytest

from app.tools.sql import SQLTool, SQLToolError


@pytest.fixture
def database(tmp_path):
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)

    connection.execute(
        """
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            salary INTEGER
        )
        """
    )

    connection.executemany(
        """
        INSERT INTO employees
        (name, department, salary)
        VALUES (?, ?, ?)
        """,
        [
            ("Alice", "Engineering", 100000),
            ("Bob", "HR", 70000),
            ("Charlie", "Engineering", 90000),
        ],
    )

    connection.commit()
    connection.close()

    return str(db_path)


@pytest.fixture
def sql_tool(database):
    return SQLTool(database=database)


def test_select_returns_rows(sql_tool):
    result = sql_tool.execute(
        "SELECT name, salary FROM employees"
    )

    assert result["row_count"] == 3
    assert result["columns"] == ["name", "salary"]


def test_where_query(sql_tool):
    result = sql_tool.execute(
        "SELECT name FROM employees WHERE department = ?",
        ("Engineering",),
    )

    assert result["row_count"] == 2


def test_parameterized_query(sql_tool):
    result = sql_tool.execute(
        "SELECT name FROM employees WHERE salary > ?",
        (80000,),
    )

    names = [row["name"] for row in result["rows"]]

    assert names == ["Alice", "Charlie"]


def test_only_select_allowed(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "UPDATE employees SET salary = 200000"
        )


def test_insert_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "INSERT INTO employees VALUES (4, 'Dave', 'IT', 80000)"
        )


def test_delete_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "DELETE FROM employees WHERE id = 1"
        )


def test_drop_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "DROP TABLE employees"
        )


def test_alter_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "ALTER TABLE employees ADD COLUMN age INTEGER"
        )


def test_empty_query_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute("")


def test_non_string_query_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(123)


def test_multiple_statements_blocked(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "SELECT * FROM employees; SELECT * FROM employees"
        )


def test_invalid_sql(sql_tool):
    with pytest.raises(SQLToolError):
        sql_tool.execute(
            "SELECT something_that_does_not_exist FROM employees"
        )


def test_max_rows(sql_tool):
    tool = SQLTool(
        database=sql_tool.database,
        max_rows=2,
    )

    result = tool.execute(
        "SELECT * FROM employees"
    )

    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_empty_result(sql_tool):
    result = sql_tool.execute(
        "SELECT * FROM employees WHERE salary > ?",
        (999999,),
    )

    assert result["row_count"] == 0
    assert result["rows"] == []


def test_invalid_max_rows():
    with pytest.raises(ValueError):
        SQLTool(max_rows=0)


def test_invalid_timeout():
    with pytest.raises(ValueError):
        SQLTool(timeout=0)
