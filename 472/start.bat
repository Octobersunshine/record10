@echo off
chcp 65001 >nul
echo ========================================
echo   Email Template Render API
echo ========================================
echo.
echo Starting service on port 8000...
echo.
echo API Documentation:
echo   Swagger UI: http://localhost:8000/docs
echo   ReDoc:      http://localhost:8000/redoc
echo.
echo Health Check: http://localhost:8000/api/health
echo.
echo ========================================
echo.
py main.py
