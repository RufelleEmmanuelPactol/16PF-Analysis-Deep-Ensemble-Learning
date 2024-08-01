from typing import Union

import mysql.connector
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def get_engine() -> Union[PooledMySQLConnection, MySQLConnectionAbstract]:
    # Create a MySQL connection using mysql.connector
    connection = mysql.connector.connect(
        host='monorail.proxy.rlwy.net',
        port=45826,
        user='root',
        password='VoUeejgBIkMgYiPmYHxMFsIXffwxCKBK',
        database='railway'
    )
    return connection
