from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(50), nullable=True) # First Name
    surname = db.Column(db.String(50), nullable=True) # Last Name
    phone = db.Column(db.String(20), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    address = db.Column(db.Text, nullable=True)
    identity_proof = db.Column(db.String(255), nullable=True) # Path to ID proof file

    email = db.Column(db.String(120), unique=True, nullable=True) # Optional for CS/Admin
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False) # 'farmer', 'cs', 'admin'
    
    # Relationships
    farmer_requests = db.relationship('ServiceRequest', foreign_keys='ServiceRequest.farmer_id', backref='farmer', lazy=True)
    cs_requests = db.relationship('ServiceRequest', foreign_keys='ServiceRequest.cs_id', backref='cs_agent', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ServiceRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(36), unique=True, nullable=False) # UUID for tracking
    farmer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    cs_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # Assigned CS agent
    status = db.Column(db.String(20), default='pending') # pending, active, closed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime, nullable=True)
    
    messages = db.relationship('ChatMessage', backref='request', lazy=True)
    feedback = db.relationship('Feedback', backref='request', uselist=False, lazy=True)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('service_request.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=True)
    file_path = db.Column(db.String(255), nullable=True) # For attachments
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    sender = db.relationship('User', foreign_keys=[sender_id])

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('service_request.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False) # 1-5
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
