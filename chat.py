from sqlalchemy.orm import Session
from .db import models

def get_policy(db: Session, district_name: str):
    policy = models.Policy(district_name=district_name)
    return db.query(models.User).filter(models.User.id == user_id).first()

