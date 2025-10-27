import sys
import os
from dotenv import load_dotenv

load_dotenv()

SQL_PASSWORD = os.getenv('MYSQL_PASSWORD')

# Add the parent directory (src/) to sys.path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.logging_config import setup_logging
import mysql.connector
from mysql.connector import Error

# Initialize logger using the setup_logging method
logger = setup_logging()

def get_db_connection():
    """
    Establish a connection to the MySQL database for GoldLens AI.
    Logs both successful and failed connection attempts.
    """
    try:
        connection = mysql.connector.connect(
            host="PraveenTak.mysql.pythonanywhere-services.com",           
            user="PraveenTak",                
            password=SQL_PASSWORD,  
            database="PraveenTak$goldlens_ai"      
        )

        if connection.is_connected():
            logger.info("Successfully connected to MySQL database: goldlens_ai")
            return connection

    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None
