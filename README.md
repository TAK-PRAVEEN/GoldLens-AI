<div align="center">
  <!-- <img src="./templates/css/Gold+Lens.gif" alt="GoldLens AI Banner" width="500"/> -->
  <!-- <img src="https://github.com/user-attachments/assets/a3252ec1-c84f-4194-8b68-c9078e09f702" alt="GoldLens AI Banner" width="500"> -->
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
| **SQLite/MySQL** | User database |
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

### Deployment
| Platform | Purpose |
|----------|---------|
| **Hugging Face Spaces** | Production hosting |
| **Docker** | Containerization |
| **Git LFS** | Large file management |

---

## 📁 Project Structure

