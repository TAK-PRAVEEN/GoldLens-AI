import os
import sys
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

SQL_HOST = os.getenv('MYSQL_HOST')
SQL_USER = os.getenv('MYSQL_USER')
SQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
SQL_DATABASE = os.getenv('MYSQL_DATABASE')
SQL_PORT = int(os.getenv('MYSQL_PORT', 3306))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from database.db_config import get_db_connection
from utils.logging_config import setup_logging

# Initialize shared logger
logger = setup_logging()


def create_users_table():
    """
    Creates 'users' table inside goldlens_ai if it does not exist.
    """
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255)
                );
                """
            )
            conn.commit()
            logger.info("Table 'users' checked/created successfully.")
        except Error as e:
            logger.error(f"Error creating 'users' table: {e}")
        finally:
            cursor.close()
            conn.close()
    else:
        logger.error("Connection to database failed. Unable to create 'users' table.")

def initialize_database():
    """
    Initialize the database and tables for GoldLens AI.
    """
    logger.info("Database initialization started.")
    create_users_table()
    logger.info("Database initialization completed.")
