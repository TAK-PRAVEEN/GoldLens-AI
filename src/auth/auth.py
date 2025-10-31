from authlib.integrations.starlette_client import OAuth
from fastapi import Request
from starlette.responses import RedirectResponse
# from dotenv import load_dotenv
import os

# Load environment variables from .env file
# load_dotenv()

# Get Google OAuth credentials from environment variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Check if credentials are loaded
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise ValueError("Google OAuth credentials not found in .env file")

# Initialize OAuth
oauth = OAuth()

# Google OAuth configuration
CONF_URL = 'https://accounts.google.com/.well-known/openid-configuration'
oauth.register(
    name='google',
    server_metadata_url=CONF_URL,
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    client_kwargs={
        'scope': 'openid email profile'
    }
)


async def google_login(request: Request):
    """
    Initiate Google OAuth login
    Redirects user to Google login page
    """
    redirect_uri = request.url_for('google_auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)


async def google_auth_callback(request: Request):
    """
    Handle Google OAuth callback
    Returns user information including email
    """
    try:
        # Get access token from Google
        token = await oauth.google.authorize_access_token(request)
        
        # Parse user information from token
        user_info = token.get('userinfo')
        
        if user_info:
            # Extract user details
            email = user_info.get('email')
            name = user_info.get('name')
            picture = user_info.get('picture')
            
            # Return user data dictionary
            return {
                'email': email,
                'name': name,
                'picture': picture,
                'authenticated': True
            }
        else:
            return {
                'authenticated': False,
                'error': 'Failed to retrieve user information'
            }
            
    except Exception as e:
        return {
            'authenticated': False,
            'error': str(e)
        }
