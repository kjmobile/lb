import pymysql
import sqlite3
import pandas as pd

def backup_mysql_to_sqlite():
    # MySQL 연결
    mysql_conn = pymysql.connect(
        host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
        user='erau',
        password='1212',
        database='airline_db',
        charset='utf8mb4'
    )
    
    # SQLite 연결 (파일이 없으면 자동 생성)
    sqlite_conn = sqlite3.connect('airline_db.sqlite')
    
    # 모든 테이블 목록 가져오기
    cursor = mysql_conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"Found {len(tables)} tables to backup:")
    for table in tables:
        print(f"- {table}")
    
    # 각 테이블 백업
    for table_name in tables:
        try:
            print(f"Backing up table: {table_name}")
            
            # MySQL에서 데이터 읽기
            df = pd.read_sql(f"SELECT * FROM {table_name}", mysql_conn)
            
            # SQLite에 저장
            df.to_sql(table_name, sqlite_conn, index=False, if_exists='replace')
            
            print(f"✓ {table_name}: {len(df)} rows copied")
            
        except Exception as e:
            print(f"✗ Error backing up {table_name}: {e}")
    
    # 연결 종료
    mysql_conn.close()
    sqlite_conn.close()
    
    print("\nBackup completed! File saved as: airline_db.sqlite")
    print("You can open this file with DB Browser for SQLite or any SQLite client.")

if __name__ == "__main__":
    backup_mysql_to_sqlite()