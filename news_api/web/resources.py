from flask import render_template
from . import web_bp






@web_bp.route('/test')
def index():
   
    data = [{}]

    return render_template('index.html', data=data)