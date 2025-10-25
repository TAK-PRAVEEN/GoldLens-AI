from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add session middleware
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Import modules
from src.database import user_crud
from src.auth.auth import google_login, google_auth_callback, oauth
from src.utils.prediction_utils import predict_future_prices, get_historical_data

# Mount static files
app.mount("/css", StaticFiles(directory=str(TEMPLATES_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(TEMPLATES_DIR / "js")), name="js")

# Configure Jinja2 templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Prediction Request Model
class PredictionRequest(BaseModel):
    model: str
    range: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        user = user_crud.authenticate_user(email, password)
        
        if user:
            request.session['user_email'] = email
            return RedirectResponse(url="/predictions", status_code=303)
        else:
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": "Invalid email or password"
                }
            )
    except Exception as e:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": f"Login failed: {str(e)}"
            }
        )


@app.get("/login/google")
async def login_google(request: Request):
    return await google_login(request)


@app.get("/auth", name="google_auth_callback")
async def auth(request: Request):
    user_data = await google_auth_callback(request)
    
    if user_data.get('authenticated'):
        email = user_data.get('email')
        existing_user = user_crud.get_user_by_email(email)
        
        if not existing_user:
            user_crud.register_google_user(email)
        
        request.session['user_email'] = email
        return RedirectResponse(url="/predictions", status_code=303)
    else:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": f"Google authentication failed: {user_data.get('error')}"
            }
        )


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        existing_user = user_crud.get_user_by_email(email)
        if existing_user:
            return templates.TemplateResponse(
                "register.html",
                {
                    "request": request,
                    "error": "User already exists with this email"
                }
            )
        
        user_crud.register_user(email, password)
        request.session['user_email'] = email
        return RedirectResponse(url="/predictions", status_code=303)
    except Exception as e:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": f"Registration failed: {str(e)}"
            }
        )


@app.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request):
    user_email = request.session.get('user_email')
    if not user_email:
        return RedirectResponse(url="/login", status_code=303)
    
    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "user_email": user_email
        }
    )


# New API endpoint for predictions
@app.post("/api/predict")
async def api_predict(request: Request, prediction_data: PredictionRequest):
    """
    API endpoint to get predictions
    """
    # Check if user is logged in
    user_email = request.session.get('user_email')
    if not user_email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Map range to number of days
    range_map = {
        "1d": 1,
        "1w": 7,
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365
    }
    
    n_days = range_map.get(prediction_data.range)
    if not n_days:
        raise HTTPException(status_code=400, detail="Invalid range")
    
    # Get predictions
    result = predict_future_prices(prediction_data.model, n_days)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Add historical data for context
    historical = get_historical_data(days=90)
    result["historical"] = historical
    
    return JSONResponse(content=result)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
