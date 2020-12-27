import sqlite3
from db import db

class UserModel(db.Model):
    # This table will be created to save all instances of UserModel
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    fname = db.Column(db.String(80))
    pan = db.Column(db.String(80))
    date = db.Column(db.String(80))

    def __init__(self, name=None, fname=None, pan=None, date=None):
        self.name = name
        self.fname = fname
        self.pan = pan
        self.date = date

    def json(self):
        """
        Return JSON representation of the user object
        """
        return {'Name':self.name, 'Fathers Name':self.fname,
                'PAN': self.pan, 'Date':self.date}

    def save_to_db(self):
        """
        Saves User object to the database
        """
        db.session.add(self)
        db.session.commit()
