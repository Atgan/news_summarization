import os
from celery.schedules import crontab
import datetime







#scrapy crawl -s MONGODB_URI="mongodb://localhost:27017/news_article" -s MONGODB_DATABASE="news_articles" scraping




class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    MONGO_URI = "mongodb://localhost:27017/news_scraping"
    JWT_SECRET_KEY = "SECRET-o"
    JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(hours=1)
    CELERY_BROKER_URL = 'amqp://guest:guest@localhost/'
    CELERY_RESULT_BACKEND = 'rpc://'
    CELERYBEAT_SCHEDULE = {
    "summarize_the_news": {
        "task": "news_api.tasks.resources.summarize_news",
        "schedule": crontab(minute=1)
    }
    
 } 


'''
nohup celery -A smartwaterapi.make_celery beat --loglevel=info > beat.log 2>&1 &
nohup celery -A smartwaterapi.make_celery worker --loglevel=info > worker.log 2>&1 &
nohup python3 run.py >> app.log 2>&1 &

'''


