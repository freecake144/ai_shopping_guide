import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.main import db, User, InteractionTurn, ExperimentSession  # 注意路径：如果 models/main.py 是你的模型文件
import json
from datetime import datetime

# ==================== 配置 ====================
# 数据库路径（与 app.py 一致）
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'experiment.db')
ENGINE = create_engine(f'sqlite:///{DB_PATH}')

# 输出目录
EXPORT_DIR = os.path.join(BASE_DIR, 'data_export')
os.makedirs(EXPORT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# =============================================

def flatten_json(df, json_cols):
    """将 JSON 列展开为多列（方便 Excel 查看）"""
    for col in json_cols:
        if col in df.columns:
            # JSON 列转为 DataFrame 并合并
            json_df = pd.json_normalize(df[col])
            json_df.columns = [f"{col}_{subcol}" for subcol in json_df.columns]
            df = pd.concat([df.drop(columns=[col]), json_df], axis=1)
    return df


def export_sessions():
    """导出会话元数据"""
    query = pd.read_sql_query("SELECT * FROM experiment_session", ENGINE)
    query['start_time'] = pd.to_datetime(query['start_time'])
    query['end_time'] = pd.to_datetime(query['end_time'])
    outfile = os.path.join(EXPORT_DIR, f'sessions_{TIMESTAMP}.csv')
    query.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"会话数据导出完成: {outfile} ({len(query)} 条)")


def export_users():
    """导出用户表"""
    query = pd.read_sql_query("SELECT * FROM users", ENGINE)
    query['created_at'] = pd.to_datetime(query['created_at'])
    outfile = os.path.join(EXPORT_DIR, f'users_{TIMESTAMP}.csv')
    query.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"用户数据导出完成: {outfile} ({len(query)} 条)")


def export_turns():
    """导出交互轮次（核心过程数据）"""
    # 先读取所有数据
    query = pd.read_sql_query("SELECT * FROM interaction_turns", ENGINE)
    query['timestamp'] = pd.to_datetime(query['timestamp'])

    # 处理 JSON 列：展开 preference_vector 和 recommended_products
    json_cols = ['preference_vector', 'recommended_products']
    query = flatten_json(query, json_cols)

    outfile = os.path.join(EXPORT_DIR, f'turns_{TIMESTAMP}.csv')
    query.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"交互轮次数据导出完成: {outfile} ({len(query)} 条)")


def export_full_joined():
    """导出合并表：每轮交互 + 会话分组信息（最常用，用于后续分析）"""
    turns_sql = """
    SELECT 
        it.*,
        es.group_id,
        es.assigned_adaptivity,
        es.assigned_calibration,
        es.start_time AS session_start_time,
        es.end_time AS session_end_time
    FROM interaction_turns it
    LEFT JOIN experiment_session es ON it.session_uuid = es.session_uuid
    """
    df = pd.read_sql_query(turns_sql, ENGINE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['session_start_time'] = pd.to_datetime(df['session_start_time'])
    df['session_end_time'] = pd.to_datetime(df['session_end_time'])

    # 展开 JSON
    df = flatten_json(df, ['preference_vector', 'recommended_products'])

    outfile = os.path.join(EXPORT_DIR, f'full_joined_data_{TIMESTAMP}.csv')
    df.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"完整合并数据导出完成: {outfile} ({len(df)} 条) - 推荐用于论文分析")


def main():
    print("开始导出实验数据...")
    export_users()
    export_sessions()
    export_turns()
    export_full_joined()
    print(f"\n所有数据已导出到文件夹: {EXPORT_DIR}")
    print("提示：")
    print("1. full_joined_data_*.csv 是最常用的（包含分组、偏好向量、drift、推荐商品等）")
    print("2. recommended_products 已展开为多列（如 recommended_products_0_product_id）")
    print("3. 可直接用 pandas/Stata/SPSS 打开进行序列挖掘、时间序列分析、SEM 等")


if __name__ == '__main__':
    main()