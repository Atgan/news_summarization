import os
from news_api import celery_init_app, create_app





app = create_app()
celery = celery_init_app(app)





'''if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)'''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    






