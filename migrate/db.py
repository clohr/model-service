from contextlib import contextmanager

from sqlalchemy import Boolean, Column, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import FLOAT, INT8RANGE
from sqlalchemy.engine import reflection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.schema import MetaData


def get_dataset_table(schema):

    Base = declarative_base(metadata=MetaData(schema=schema))

    class DatasetTable(Base):
        __tablename__ = "datasets"

        id = Column(Integer, primary_key=True)
        node_id = Column(String)
        name = Column(String)
        # locked = Column(Boolean)
        state = Column(String)

        def __repr__(self):
            return f"<Dataset name={self.name}, id={self.id}, node_id={self.node_id}, state={self.state}>"

    return DatasetTable


def get_organizations(engine):
    def is_organization_schema(name) -> bool:
        try:
            int(name)
            return True
        except:
            return False

    insp = reflection.Inspector.from_engine(engine)
    organizations = [
        int(name) for name in insp.get_schema_names() if is_organization_schema(name)
    ]
    organizations.sort()
    return organizations


@contextmanager
def session_scope(Session):
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def get_postgres(connection_string):
    engine = create_engine(connection_string)
    session_factory = sessionmaker(bind=engine)
    return engine, session_factory
