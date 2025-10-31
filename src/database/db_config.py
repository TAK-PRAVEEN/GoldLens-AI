import sys
import os
from dotenv import load_dotenv

load_dotenv()

SQL_HOST = os.getenv('MYSQL_HOST')
SQL_USER = os.getenv('MYSQL_USER')
SQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
SQL_DATABASE = os.getenv('MYSQL_DATABASE')
SQL_PORT = int(os.getenv('MYSQL_PORT', 3306))  # Default to 3306 if not set

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.logging_config import setup_logging
import mysql.connector
from mysql.connector import Error

logger = setup_logging()

def get_db_connection():
    """
    Establish a connection to the MySQL database using FreeSQLDatabase.com credentials.
    Logs both successful and failed connection attempts.
    """
    try:
        connection = mysql.connector.connect(
            host=SQL_HOST,
            user=SQL_USER,
            password=SQL_PASSWORD,
            database=SQL_DATABASE,
            port=SQL_PORT
        )

        if connection.is_connected():
            logger.info(f"Successfully connected to MySQL database: {SQL_DATABASE} on {SQL_HOST}")
            return connection

    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None
