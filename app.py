import os
from flask import Flask, render_template, request, session, redirect, url_for
from flask_bootstrap import Bootstrap
from forms import PANUploadForm, EditForm
from models.user import UserModel
from db import db
import cv2
import numpy as np
import pan

app = Flask(__name__)

# Using Bootstrap as base template for easy styling
bootstrap = Bootstrap(app)

# Secret key for Flask-WTF Forms
app.config['SECRET_KEY'] = 'LENDENCLUB ASSIGNMENT' # Replace with os.environ['SECRET_KEY'] in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create an empty table if no table exists
@app.before_first_request
def create_tables():
    db.create_all()

# Home page will have a simple form with a file Upload Field
@app.route('/', methods=['GET', 'POST'])
def home():
    form = PANUploadForm()
    if form.validate_on_submit():
        # If user has successfully uploaded an image, converting into an OpenCV image
        # The file is being directly read from memory instead of disk
        image = cv2.imdecode(np.frombuffer(form.file.data.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # Image will be processed to extract the necessary info
        extracted_info = pan.extract_info(image)
        # Saving extracted info into a session object so that it is availabe to other functions
        session['extracted_info'] = extracted_info
        # Redirecting user to another page, where they can correct the information extracted
        return redirect(url_for('add_user_info'))

    return render_template('home.html', form=form)

# This page will show extracted information from the image and let the user edit and submit it
@app.route('/add_info', methods=['GET', 'POST'])
def add_user_info():
    form = EditForm()

    # Pre-populating the form
    if request.method == 'GET':
        extracted_info = {'name':None, 'fname':None,
                          'pan': None, 'date': None}
        extracted_info.update(session['extracted_info'])
        form.name.data = extracted_info['name']
        form.fname.data = extracted_info['fname']
        form.pan.data = extracted_info['pan']
        form.date.data = extracted_info['date']

    if form.validate_on_submit():
        # If user successfuly fills the information, his details will be saved.
        new_user = UserModel()
        form.populate_obj(new_user)
        new_user.save_to_db()
        return render_template('thanks.html')

    return render_template('user_form.html', form=form)

@app.route('/users_977')
def get_users():
    return {'Users': [user.json() for user in UserModel.query.all()]}

if __name__ == '__main__':
    db.init_app(app)
    app.run(port=5000, debug=True)
