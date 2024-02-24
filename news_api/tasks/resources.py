from celery import shared_task
import logging
from ..nlp_summarization.nlp_copy import Summarizer
from news_api.nlp_summarization.schema import NewsDocument, NewsDocumentSchema
from flask import jsonify
from . import tasks_bp
import subprocess
import os









#@tasks_bp.route("/do")
@shared_task
def summarize_news():
    summarizer = Summarizer()

    news_summarize = summarizer.summarize_and_update_db()

    return news_summarize






@shared_task
def run_scrapy_crawler():
    
    project_directory = "/Users/erickndoho/Documents/Engineering/ai_news_summary/news_api/news_scraping/cnn_crawler/spiders"
    os.chdir(project_directory)
    
    command = "scrapy crawl -s MONGODB_URI='mongodb://localhost:27017/news_scraping' -s MONGODB_DATABASE='news_scraping' cnn_crawler"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except FileNotFoundError:
        print("Error: 'scrapy' command not found. Make sure Scrapy is installed and added to the PATH.")




