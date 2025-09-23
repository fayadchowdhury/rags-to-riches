from typing import Dict
from dotenv import load_dotenv
import os

from core.database.BaseDatabase import BaseDatabase
from core.database.SQLiteDatabase import SQLiteDatabase

def get_app_env_config(path: str) -> Dict:
    '''
    Take path to specified .env file
    Return dictionary of environment variables
    '''
    load_dotenv(path)
    return {
        "DB_HOST": os.environ.get("DB_HOST", ""),
        "DB_PORT": os.environ.get("DB_PORT", ""),
        "DB_DATABASE": os.environ.get("DB_DATABASE", ""),
        "DB_USERNAME": os.environ.get("DB_USERNAME", ""),
        "DB_PASSWORD": os.environ.get("DB_PASSWORD", ""),
    }

def initialize_database(db_config: Dict) -> BaseDatabase:
    '''
    Take a type and database config
    Return a database
    '''
    database = None
    type = db_config.get("type", "")
    if type == "SQLiteDatabase":
        database = SQLiteDatabase(**db_config)

    return database