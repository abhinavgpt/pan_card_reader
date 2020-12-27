from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired

# Form for image upload
class PANUploadForm(FlaskForm):
    file = FileField('PAN card image', validators=[
                                  FileRequired(),
                                  FileAllowed(['jpg', 'png', 'jpeg'], # Only these extendions will be allowed
                                  'Images only! (JPG, JPEG, PNG)')])
    submit = SubmitField('Upload')

# Form for user information edit and upload
class EditForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    fname = StringField("Father's Name", validators=[DataRequired()])
    pan = StringField('PAN', validators=[DataRequired()])
    date = StringField('Date', validators=[DataRequired()])
    submit = SubmitField('Submit')
