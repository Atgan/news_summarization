
from mongoengine import Document, StringField
from marshmallow import Schema, fields, validate





class NewsDocument(Document):
    content = StringField()
    date_written = StringField()
    authors = StringField()
    summary = StringField()
    summary_date = StringField()




class NewsDocumentSchema(Schema):
    content = fields.String()
    date_written = fields.String()
    authors = fields.String()
    summary = fields.Str(required=True)
    summary_date = fields.String()