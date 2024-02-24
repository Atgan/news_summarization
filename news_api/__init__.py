from flask import Flask, jsonify, render_template
from flask_pymongo import PyMongo
from .config import Config
from flask_jwt_extended import JWTManager
import mongoengine
import logging
from celery import Celery
from celery import Task
from flask_cors import CORS
from .news_scraping.cnn_crawler.items import *
from news_api.tasks.resources import summarize_news
from .nlp_summarization.schema import NewsDocument










logging.basicConfig(level=logging.DEBUG) 
logger = logging.getLogger(__name__)
app = Flask(__name__)




@app.route('/', methods=['GET'])
def api_check():
    return jsonify({"status": "success", "message": "API is online"}), 200


@app.route('/test')
def index():
   
    data = NewsDocument.objects.all()

    return render_template('index.html', data=data)




def create_app():
    app.config.from_object(Config)
    app.debug = True
    mongo = PyMongo(app)
    mongoengine.connect(host=app.config['MONGO_URI'])
    app.config.from_mapping(
        CELERY=dict(
            broker_url=app.config['CELERY_BROKER_URL'],
            result_backend=app.config['CELERY_RESULT_BACKEND'],
            task_ignore_result=True,
        ),
    )
    celery = celery_init_app(app)
    celery.conf.update(app.config)
    CORS(app)


    from .tasks import tasks_bp
    from .web import web_bp


    app.register_blueprint(tasks_bp)
    app.register_blueprint(web_bp)


    return app


def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery = Celery(app.name, task_cls=FlaskTask)
    celery.config_from_object(app.config["CELERY"])
    celery.set_default()
    app.extensions["celery"] = celery
    
    return celery


