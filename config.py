import os
from dotenv import load_dotenv
#加载环境变量
load_dotenv()

#数据库配置
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "data/experiment.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

#deepseek配置
DEEPSEEK_API_KEY = os.getenv("sk-8199ad0741d842d5a0ff2cdc19fdd8c8")