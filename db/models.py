from sqlalchemy import Column, String, TEXT, INT, BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Policy(Base):
    __tablename__ = "policy"

    district_name = Column(String(5), nullable=False, primary_key=True)
    contents = Column(TEXT, nullable=False)

class Admin(Base):
    __tablename__ = "admin"

    id = Column(String(15), nullable=False, primary_key=True) 
    password = Column(String(30), nullable=False) 

class LargeWaste(Base):
    __tablename__ = "large_waste"

    id = Column(INT, nullable=False, primary_key=True)
    district_name = Column(String(5), nullable=False)
    large_waste = Column(String(20), nullable=False)
    policy = Column(TEXT, nullable=False)