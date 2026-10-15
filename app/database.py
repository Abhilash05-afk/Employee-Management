import psycopg2


DATABASE_CONFIG = {
    "host": "127.0.0.1",
    "database": "employee_attrition",
    "user": "postgres",
    "password": "2002",
    "port": 5432
}


def get_connection():
    return psycopg2.connect(**DATABASE_CONFIG)