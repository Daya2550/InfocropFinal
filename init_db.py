from app import app, db
from models import User

with app.app_context():
    try:
        db.create_all()
        print("Database tables created successfully.")
        
        # Create default admin if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Default admin created (admin/admin123).")
        else:
            print("Admin already exists.")
            
    except Exception as e:
        print(f"Error initializing database: {e}")
