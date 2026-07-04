#!/usr/bin/env python3
import argparse
import sqlite3
import os

parser = argparse.ArgumentParser(description='Dump SQLite DB schema (tables, indexes, triggers, views) to SQL file')
parser.add_argument('db', nargs='+', help='Path(s) to sqlite database file(s)')
parser.add_argument('-o', '--out', help='Output SQL file (if single DB)')
args = parser.parse_args()

for db_path in args.db:
    db_path = os.path.expanduser(db_path)
    if not os.path.isfile(db_path):
        print(f"ERROR: no existe el archivo {db_path}")
        continue
    base = os.path.basename(os.path.dirname(db_path)) or 'db'
    name = os.path.splitext(os.path.basename(db_path))[0]
    if args.out and len(args.db) == 1:
        out_path = args.out
    else:
        out_path = os.path.join(os.getcwd(), f"{base}_{name}_schema.sql")
    print(f"Generando esquema de {db_path} a {out_path}...")
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("""
            SELECT type, name, sql
            FROM sqlite_master
            WHERE sql IS NOT NULL
              AND name NOT LIKE 'sqlite_%'
              AND type IN ('table','index','trigger','view')
            ORDER BY CASE type WHEN 'table' THEN 1 WHEN 'view' THEN 2 WHEN 'index' THEN 3 WHEN 'trigger' THEN 4 END, name
        """)
        rows = cur.fetchall()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"-- Schema dump for: {db_path}\n")
            f.write("PRAGMA foreign_keys=OFF;\n")
            f.write("BEGIN TRANSACTION;\n\n")
            for r in rows:
                sql = r[2].strip()
                if not sql.endswith(';'):
                    sql = sql + ';'
                f.write(sql + '\n\n')
            f.write("COMMIT;\n")
        con.close()
        print(f"Esquema generado: {out_path}")
    except Exception as e:
        print(f"Fallo al generar esquema {db_path}: {e}")
