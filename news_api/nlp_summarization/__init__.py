from flask import Blueprint

summarize_bp = Blueprint("summarize", __name__)


from . import summarize_bp