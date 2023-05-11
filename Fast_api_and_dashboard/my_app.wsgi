from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from uvicorn import run

from main import app