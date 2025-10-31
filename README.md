<div align="center">
  <img src="https://github.com/user-attachments/assets/a2ae38d8-e58e-446a-91c5-6ac7a96155ac" alt="GoldLens AI Banner" width="500">

  **AI-Powered Gold Price Prediction & Real-Time Market Dashboard**
  
  [![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/PraveenTak/GoldLens-AI)
  [![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
  [![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Preview](#-preview)
- [Features](#-features)
- [Quick Links](#-quick-links)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Models](#-models)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

**GoldLens-AI** is a full-stack ML application that provides:
- **Time-series forecasting** of gold prices using LSTM, GRU, and ensemble models
- **Real-time gold prices** across multiple currencies (USD, EUR, GBP, INR - 18K/22K/24K)
- **AI-generated motivational quotes** via Google Gemini API
- **User authentication** with Google OAuth and custom login
- **Interactive web dashboard** for visualization and predictions

Perfect for traders, analysts, and anyone interested in gold market trends!

---

## 📸 Preview
<img width="1903" height="1076" alt="Screenshot 2025-10-31 230104" src="https://github.com/user-attachments/assets/803e3cf0-406b-4e04-bc36-b881c206a9f1" />
<img width="1915" height="1079" alt="Screenshot 2025-10-31 230145" src="https://github.com/user-attachments/assets/50caa17d-41da-455d-91ca-a4e7eaab0fd8" />
<img width="1917" height="1079" alt="Screenshot 2025-10-31 230214" src="https://github.com/user-attachments/assets/356d24e3-6967-4b9e-b12e-07b39306ae84" />
<img width="1917" height="1075" alt="Screenshot 2025-10-31 230254" src="https://github.com/user-attachments/assets/d2bb328a-34a4-4d8e-be82-2026a2f2b664" />
<img width="1919" height="1079" alt="Screenshot 2025-10-31 230402" src="https://github.com/user-attachments/assets/0a96daaa-b916-4b89-82cf-188cfef7c894" />
<img width="998" height="502" alt="Screenshot 2025-10-31 230444" src="https://github.com/user-attachments/assets/f8c1646a-0cb8-4b13-b6ef-20239177892d" />

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Price Prediction** | Forecast gold prices for next day/week/month/year |
| 💱 **Multi-Currency Support** | Real-time rates in USD, EUR, GBP, INR (multiple purities) |
| 🤖 **AI Quotes** | Daily motivational quotes powered by Google Gemini |
| 🔐 **Authentication** | Google OAuth + custom email/password login |
| 📊 **Interactive Charts** | Visualize historical data and predictions |
| ⚡ **Fast API** | RESTful endpoints for easy integration |
| 🎨 **Responsive UI** | Beautiful, mobile-friendly web interface |

---

## 🔗 Quick Links

| Resource | URL |
|----------|-----|
| 🚀 **Live Demo** | [Hugging Face Space](https://huggingface.co/spaces/PraveenTak/GoldLens-AI) |
| 📂 **GitHub Repo** | [TAK-PRAVEEN/GoldLens-AI](https://github.com/TAK-PRAVEEN/GoldLens-AI) |
| 📊 **Notebooks** | [Exploratory Analysis](./Notebooks/) |
| 🎨 **Frontend** | [Templates & CSS](./templates/) |
| 🧠 **Models** | [Trained Models](./models/) |
| 📈 **Data** | [Gold Price Datasets](./data/) |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core programming language |
| **FastAPI** | High-performance web framework |
| **Uvicorn** | ASGI server |
| **MySQL** | User database |
| **python-dotenv** | Environment variable management |

### Machine Learning
| Library | Use Case |
|---------|----------|
| **TensorFlow/Keras** | LSTM, GRU models |
| **scikit-learn** | Data preprocessing, metrics |
| **pandas** | Data manipulation |
| **numpy** | Numerical operations |

### Frontend
| Tech | Description |
|------|-------------|
| **Jinja2** | Server-side templating |
| **HTML5/CSS3** | Responsive UI |
| **JavaScript** | Client-side interactivity |
| **Chart.js** | Data visualization |

### APIs & Services
| Service | Integration |
|---------|-------------|
| **Google Gemini API** | AI-generated quotes |
| **Gold Price API** | Real-time market data |
| **Google OAuth** | User authentication |
| **infinityfree.com** | MySQL Database |

### Deployment
| Platform | Purpose |
|----------|---------|
| **Hugging Face Spaces** | Production hosting |
| **Docker** | Containerization |
| **Git LFS** | Large file management |

---

## 📁 Project Structure
```bash
GoldLens-AI/
│
├── data/ # Datasets
│ ├── processed/ # Feature-engineered data
│ └── raw/ # Original gold price data
│ └── gold_daily.csv
│
├── models/ # Trained ML models
│ ├── bilstm_best.keras # BiLSTM model
│ ├── lstm_best.keras # LSTM model
│ ├── gru_best.keras # GRU model
│ ├── ensemble.keras # Ensemble model
│ ├── scaler.pkl # Data scaler
│ └── metrics.json # Model performance metrics
│
├── Notebooks/ # Jupyter notebooks
│ └── (EDA & model experiments)
│
├── src/ # Source code
│ ├── api/ # FastAPI application
│ │ └── app.py # Main API routes
│ ├── auth/ # Authentication logic
│ │ └── auth.py # OAuth & custom login
│ ├── database/ # Database config
│ │ ├── db_config.py # MySQL connection
│ │ └── user_crud.py # User CRUD operations
│ ├── features/ # Feature engineering
│ │ └── featurize.py
│ ├── ingest/ # Data ingestion
│ │ └── fetch.py # API data fetchers
│ ├── models/ # Model architectures
│ │ ├── architectures.py # LSTM/GRU definitions
│ │ ├── ensemble.py # Ensemble logic
│ │ └── train.py # Training scripts
│ └── utils/ # Utility functions
│ ├── logging_config.py # Logging setup
│ ├── plot_utils.py # Visualization helpers
│ └── prediction_utils.py # Prediction functions
│
├── templates/ # HTML templates
│ ├── css/ # Stylesheets & assets
│ │ ├── styles.css
│ │ ├── Gold+Lens.gif # Banner animation
│ │ ├── logo.png
│ │ └── ...
│ ├── js/ # JavaScript files
│ │ └── (client-side logic)
│ ├── home.html # Landing page
│ ├── login.html # Login page
│ ├── register.html # Registration page
│ └── predictions.html # Predictions dashboard
│
├── .env # Environment variables (not in repo)
├── .gitignore # Git ignore rules
├── .gitattributes # Git LFS tracking
├── Dockerfile # Docker container config
├── goldlens.log # Application logs
├── README.md # This file
└── requirements.txt # Python dependencies
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- Git
- (Optional) Docker

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/TAK-PRAVEEN/GoldLens-AI.git
cd GoldLens-AI
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file:

```
SECRET_KEY=your-secret-key-here
GOOGLE_API_KEY=your-gemini-api-key
GOOGLE_CLIENT_ID=your-oauth-client-id
GOOGLE_CLIENT_SECRET=your-oauth-client-secret
DB_HOST=your-db-host
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=your-db-name
```

5. **Run the application**

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 7860
```

6. **Access the app**
Open browser: `http://localhost:7860`

---

## 🚀 Usage

### Web Interface
1. Navigate to the homepage
2. View live gold prices and daily quote
3. Register/Login to access predictions
4. Select model and time range for forecasting
5. View interactive prediction charts

### API Usage

**Get Predictions:**
```bash
curl -X POST "http://localhost:7860/api/predict"
-H "Content-Type: application/json"
-d '{"model": "lstm_best", "range": "1w"}'
```

**Response:**
```bash
{
"n_days": 7,
"model": "lstm_best",
"dates": ["2025-11-01", "2025-11-02", ...],
"predictions": [2800.45, 2805.32, ...],
"confidence_lower": [2750.12, ...],
"confidence_upper": [2850.78, ...]
}
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page |
| `GET` | `/login` | Login page |
| `POST` | `/login` | Authenticate user |
| `GET` | `/register` | Registration page |
| `POST` | `/register` | Create new user |
| `GET` | `/login/google` | Google OAuth login |
| `GET` | `/auth` | OAuth callback |
| `GET` | `/predictions` | Predictions dashboard (auth required) |
| `POST` | `/api/predict` | Get price predictions |

---

## 🧠 Models

| Model | Architecture | Performance (RMSE) |
|-------|--------------|-------------------|
| **LSTM** | 3-layer LSTM with dropout | ~15.2 |
| **BiLSTM** | Bidirectional LSTM | ~14.8 |
| **GRU** | 3-layer GRU | ~15.5 |
| **Ensemble** | Weighted average of above | ~13.9 |

*Note: Performance metrics on test set (2023-2024 data)*

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Praveen Tak**

[![Portfolio](https://img.shields.io/badge/Portfolio-Website-blue)](https://tak-praveen.github.io/PraveenTak_Portfolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5)](https://www.linkedin.com/in/praveentak/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717)](https://github.com/TAK-PRAVEEN)
[![Email](https://img.shields.io/badge/Email-Contact-D14836)](mailto:praveentak715@gmail.com)

---

<div align="center">
  <p>Made with ❤️ by Praveen Tak</p>
  <p>⭐ Star this repo if you find it useful!</p>
</div>

