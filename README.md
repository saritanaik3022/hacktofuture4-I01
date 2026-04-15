The Specter Sentinals-I01
PashuMitra

Problem
Farmers often miss the 12–18 hour heat detection window, resulting in low conception rates (30–35%). Existing solutions require internet, lack local language support, and are not user-friendly for rural farmers.

Solution
PashuMitra is an offline AI-powered mobile application that helps farmers detect heat cycles and health risks using image analysis and symptom-based inputs.

Features
Image-based estrus detection
9-question symptom analysis
Hybrid AI (MobileNetV2 + XGBoost + Rule Engine)
Offline-first functionality
Auto-sync when internet is available
Multi-language support (Kannada, Hindi, Tamil, Telugu, English)
Multi-species support (cow, buffalo, goat, sheep)
Heat prediction output (Likely Heat, Monitor Closely, Healthy)
AI technician booking and scheduling
Streamlit dashboard for analytics and monitoring
Local storage using SQLite

Tech Stack
Frontend: Flutter
Backend: FastAPI
Database: SQLite, PostgreSQL
AI Models: MobileNetV2, XGBoost
Tools: TensorFlow, OpenCV, Pillow, Streamlit
