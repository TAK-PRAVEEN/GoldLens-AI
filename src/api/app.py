import logging
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
from src.utils.logging_config import setup_logging
from src.database import user_crud
from src.auth.auth import google_login, google_auth_callback, oauth
from src.utils.plot_utils import build_gold_chart
from src.utils.prediction_utils import get_historical_data, predict_future_prices

logger = setup_logging()
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
    logger.info("Rendering home page")
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    logger.info("Rendering login page")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    logger.info(f"Login attempt for email: {email}")
    try:
        user = user_crud.authenticate_user(email, password)
        if user:
            logger.info(f"User logged in: {email}")
            request.session['user_email'] = email
            return RedirectResponse(url="/predictions", status_code=303)
        else:
            logger.warning(f"Invalid login for email: {email}")
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": "Invalid email or password"
                }
            )
    except Exception as e:
        logger.error(f"Login error for {email}: {e}", exc_info=True)
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": f"Login failed: {str(e)}"
            }
        )

@app.get("/login/google")
async def login_google(request: Request):
    logger.info("Google login initiated")
    return await google_login(request)

@app.get("/auth", name="google_auth_callback")
async def auth(request: Request):
    logger.info("Google auth callback triggered")
    user_data = await google_auth_callback(request)
    if user_data.get('authenticated'):
        email = user_data.get('email')
        logger.info(f"Google auth successful for: {email}")
        existing_user = user_crud.get_user_by_email(email)
        if not existing_user:
            user_crud.register_google_user(email)
            logger.info(f"Registered new Google user: {email}")
        request.session['user_email'] = email
        return RedirectResponse(url="/predictions", status_code=303)
    else:
        logger.warning("Google authentication failed")
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": f"Google authentication failed: {user_data.get('error')}"
            }
        )

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    logger.info("Rendering registration page")
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request, email: str = Form(...), password: str = Form(...)):
    logger.info(f"Registration attempt for {email}")
    try:
        existing_user = user_crud.get_user_by_email(email)
        if existing_user:
            logger.warning(f"Attempt to register existing user: {email}")
            return templates.TemplateResponse(
                "register.html",
                {
                    "request": request,
                    "error": "User already exists with this email"
                }
            )
        user_crud.register_user(email, password)
        request.session['user_email'] = email
        logger.info(f"User registered successfully: {email}")
        return RedirectResponse(url="/predictions", status_code=303)
    except Exception as e:
        logger.error(f"Registration error for {email}: {e}", exc_info=True)
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
    logger.info(f"Predictions page access by: {user_email}")
    if not user_email:
        logger.warning("Unauthenticated predictions page access")
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "user_email": user_email
        }
    )

@app.post("/api/predict")
async def api_predict(request: Request):
    try:
        body = await request.json()
        model_selected = body.get("model")
        range_selected = body.get("range")
        logger.info(f"Prediction requested. Model: {model_selected}, Range: {range_selected}")

        range_map = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
        n_days = range_map.get(range_selected)
        if not n_days:
            logger.warning(f"Invalid prediction range requested: {range_selected}")
            raise HTTPException(status_code=400, detail="Invalid range")

        result = predict_future_prices(model_selected, n_days)
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        logger.info(f"Prediction successful for model: {model_selected}, days: {n_days}")

        historical = get_historical_data(days=120 if n_days < 120 else n_days)

        response = {
            "n_days": n_days,
            "model": model_selected,
            "dates": result["dates"],
            "predictions": result["predictions"],
            "confidence_lower": result.get("confidence_lower", []),
            "confidence_upper": result.get("confidence_upper", []),
            "historical": historical,
        }
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Prediction API error: {e}", exc_info=True)
        import traceback
        return JSONResponse(content={"error": str(e), "traceback": traceback.format_exc()}, status_code=500)

# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting FastAPI/Gunicorn server")
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
