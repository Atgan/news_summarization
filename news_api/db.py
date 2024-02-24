from flask_pymongo import PyMongo
from celery import Celery

mongo = PyMongo()
celery = Celery()

def init_db(app):
    mongo.init_app(app)
    celery.config_from_object(app.config)

    

