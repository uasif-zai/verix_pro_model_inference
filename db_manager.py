from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy without app context
db = SQLAlchemy()

def init_db(app):
    """Initialize the database with the Flask app."""
    try:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:AIDeveloper1!@verix-pro-database.c1060y8wwp4t.us-east-2.rds.amazonaws.com:5432/postgres'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(app)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

# Define the TestTable model
class TestTable(db.Model):
    __tablename__ = 'test_table'
    id = db.Column(db.Integer, primary_key=True)
    pdf_id = db.Column(db.Integer, nullable=False)  # Plain INTEGER, NOT NULL, no FK
    data = db.Column(JSONB)
    status = db.Column(db.String(255))

def insert_test_table_data(pdf_id, json_data, status="processed"):
    """Insert data into test_table."""
    try:
        with db.session.no_autoflush:  # Prevent premature flushes
            logger.debug(f"Inserting data: pdf_id={pdf_id}, data={json_data}, status={status}")
            new_entry = TestTable(
                pdf_id=pdf_id,
                data=json_data,
                status=status
            )
            db.session.add(new_entry)
            db.session.commit()
            logger.info(f"Inserted record with id: {new_entry.id}")
            return new_entry.id
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to insert data: {str(e)}")
        raise Exception(f"Failed to insert data: {str(e)}")