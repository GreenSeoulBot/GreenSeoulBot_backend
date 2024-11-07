from sqlalchemy import Column, TEXT, INT, BIGINT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Test(Base):
    __tablename__ = "policy"

    district_name = Column(TEXT, nullable=False, primary_key=True)
    contents = Column(TEXT, nullable=False)