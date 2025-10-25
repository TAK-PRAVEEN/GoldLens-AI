import os
import sys
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from mysql.connector import Error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.db_config import get_db_connection
from utils.logging_config import setup_logging

# Create a consistent logger shared by all modules
logger = setup_logging()


def register_user(email, password):
    """
    Registers a new user with a securely hashed password.
    Uses a parameterized query to prevent SQL injection.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection unavailable. Cannot register user.")
        return False

    cursor = conn.cursor()
    try:
        hashed_password = generate_password_hash(password)
        query = """
            INSERT INTO users (email, password_hash)
            VALUES (%s, %s)
        """
        cursor.execute(query, (email, hashed_password))
        conn.commit()
        logger.info(f"Registered new user: {email}")
        return True
    except Error as e:
        logger.error(f"MySQL error during registration for {email}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during registration for {email}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def authenticate_user(email, password):
    """
    Authenticate a user by verifying the password hash.
    Returns the user dictionary if authentication succeeds, otherwise None.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection unavailable for authentication.")
        return None

    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password_hash'], password):
            logger.info(f"User {email} authenticated successfully.")
            return user
        else:
            logger.warning(f"Authentication failed for user {email}.")
            return None
    except Error as e:
        logger.error(f"MySQL error during authentication for {email}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during authentication for {email}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def register_google_user(email):
    """
    Registers a Google OAuth user in the database.
    For OAuth users, a placeholder password is generated since they won't use it.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection unavailable. Cannot register Google user.")
        return False

    cursor = conn.cursor()
    try:
        # Generate a secure random placeholder password for OAuth users
        # They won't use it since they login via Google
        import secrets
        placeholder_password = generate_password_hash(f"OAUTH_{secrets.token_hex(16)}")
        
        query = """
            INSERT INTO users (email, password_hash)
            VALUES (%s, %s)
        """
        cursor.execute(query, (email, placeholder_password))
        conn.commit()
        logger.info(f"Google OAuth user registered: {email}")
        return True
    except Error as e:
        # Check if user already exists (duplicate entry error)
        if e.errno == 1062:  # MySQL duplicate entry error code
            logger.info(f"Google user already exists: {email}")
            return True  # Return True since user exists
        else:
            logger.error(f"MySQL error during Google user registration for {email}: {e}")
            return False
    except Exception as e:
        logger.exception(f"Unhandled error during Google user registration for {email}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_user_by_email(email):
    """
    Retrieve a user by email address.
    Returns the user dictionary if found, otherwise None.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection unavailable. Cannot retrieve user.")
        return None

    cursor = conn.cursor(dictionary=True)
    try:
        query = "SELECT * FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        if user:
            logger.info(f"User found: {email}")
            return user
        else:
            logger.info(f"No user found with email: {email}")
            return None
    except Error as e:
        logger.error(f"MySQL error while retrieving user {email}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error while retrieving user {email}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()
