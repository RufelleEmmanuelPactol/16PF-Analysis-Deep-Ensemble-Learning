from typing import Union

import mysql.connector
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
import os
import dotenv

dotenv.load_dotenv()


def get_engine() -> Union[PooledMySQLConnection, MySQLConnectionAbstract]:
    # Retrieve database connection details from environment variables
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT')),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_DATABASE')
    )
    return connection
