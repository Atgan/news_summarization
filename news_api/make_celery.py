from celery import Celery, Task
from flask import Flask











from news_api import create_app

flask_app = create_app()
celery_app = flask_app.extensions["celery"]


if celery_app is None:
    raise RuntimeError("Celery extension not properly initialized")


'''def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    from .tasks.resources import send_remote_tokens_async, send_sms_async, update_meter_readings_async, monthly_reset_async
    celery.Task = ContextTask
    celery.task(send_sms_async)
    celery.task(send_remote_tokens_async)
    celery.task(update_meter_readings_async)
    celery.task(monthly_reset_async)
    app.extensions['celery'] = celery

    
    return celery'''

