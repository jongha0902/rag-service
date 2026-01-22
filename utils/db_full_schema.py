import oracledb
from utils.db import get_oracle_conn

# 1. 벡터화용 전체 스키마 추출 함수
def get_full_db_schema():
    documents = []

    # (A) 테이블 정보 조회 
    sql_tables = """
    SELECT 
        t.table_name, 
        t.comments, 
        c.column_name, 
        c.data_type, 
        cc.comments,
        t.table_type
    FROM user_tab_comments t
    JOIN user_tab_columns c ON t.table_name = c.table_name
    LEFT JOIN user_col_comments cc ON c.table_name = cc.table_name AND c.column_name = cc.column_name
    WHERE t.table_name NOT LIKE 'BIN$%'  -- 휴지통 데이터 제외
    ORDER BY t.table_name, c.column_id
    """
    
    table_map = {}
    with get_oracle_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_tables)
        for row in cursor.fetchall():
            t_name, t_cmt, c_name, d_type, c_cmt, t_type = row
            
            if t_name not in table_map:
                t_desc = f" ({t_cmt})" if t_cmt else ""
                # DB에서 가져온 실제 타입 사용
                table_map[t_name] = {
                    "header": f"Object: {t_name} (Type: {t_type}){t_desc}", 
                    "cols": [],
                    "type": t_type # 메타데이터로 쓰기 위해 저장
                }
            
            c_desc = f" -- {c_cmt}" if c_cmt else ""
            table_map[t_name]["cols"].append(f"  - {c_name} ({d_type}){c_desc}")

        for t_name, data in table_map.items():
            content = f"{data['header']}\nColumns:\n" + "\n".join(data["cols"])
            
            # 벡터 DB에 넣을 타입 결정 (TABLE / VIEW)
            doc_type = "TABLE" if data['type'] == 'TABLE' else "VIEW"
            
            documents.append({
                "content": content, 
                "type": doc_type, 
                "name": t_name
            })

    # (B) 소스 코드 (프로시저, 함수 등)
    sql_source = "SELECT name, type, text FROM user_source WHERE name NOT LIKE 'BIN$%' ORDER BY name, type, line"
    with get_oracle_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_source)
        source_map = {}
        for row in cursor.fetchall():
            name, obj_type, text = row
            key = f"{name}|{obj_type}"
            if key not in source_map: source_map[key] = []
            source_map[key].append(text.rstrip())

        for key, lines in source_map.items():
            name, obj_type = key.split("|")
            full_code = "\n".join(lines)
            content = f"Object: {name}\nType: {obj_type}\nSource Code:\n```sql\n{full_code}\n```"
            documents.append({"content": content, "type": "PROCEDURE", "name": name})

    return documents

# 2. 메타데이터 전수 조사 함수 (Hybrid Search용)
def search_db_metadata(keyword: str):
    if not keyword or len(keyword.strip()) < 2: return "키워드가 너무 짧습니다."
    
    results = []
    kw_pattern = f"%{keyword}%"
    
    with get_oracle_conn() as conn:
        cursor = conn.cursor()
        
        # (A) 테이블/컬럼 검색
        sql_tab = """
        SELECT DISTINCT t.table_name, t.comments 
        FROM user_tab_columns c
        JOIN user_tab_comments t ON c.table_name = t.table_name
        LEFT JOIN user_col_comments cc ON c.table_name = cc.table_name AND c.column_name = cc.column_name
        WHERE (upper(c.column_name) LIKE upper(:kw) 
           OR upper(cc.comments) LIKE upper(:kw) 
           OR upper(t.table_name) LIKE upper(:kw))
           AND t.table_name NOT LIKE 'BIN$%'
        """
        cursor.execute(sql_tab, kw=kw_pattern)
        rows = cursor.fetchall()
        if rows:
            results.append(f"=== [테이블] '{keyword}' 관련 검색 ===")
            for r in rows:
                results.append(f"- {r[0]} ({r[1] or ''})")
            results.append("")

        # (B) 소스코드 검색
        sql_src = """
        SELECT DISTINCT name, type FROM user_source 
        WHERE (upper(name) LIKE upper(:kw) OR upper(text) LIKE upper(:kw)) AND name NOT LIKE 'BIN$%'
        """
        cursor.execute(sql_src, kw=kw_pattern)
        rows = cursor.fetchall()
        if rows:
            results.append(f"=== [코드] '{keyword}' 관련 검색 ===")
            for r in rows:
                results.append(f"- [{r[1]}] {r[0]}")

    if not results: return "검색 결과 없음"
    return "\n".join(results)

# 3. 전체 테이블 목록 조회
def get_all_table_names():
    with get_oracle_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT table_name, comments FROM user_tab_comments WHERE table_name NOT LIKE 'BIN$%' ORDER BY table_name")
        return "\n".join([f"- {row[0]} ({row[1] or ''})" for row in cursor.fetchall()])