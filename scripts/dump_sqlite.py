#!/usr/bin/env python3
import argparse
import sqlite3
import os

parser = argparse.ArgumentParser(description='Dump SQLite DB to SQL file')
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
        out_path = os.path.join(os.getcwd(), f"{base}_{name}_dump.sql")
    print(f"Generando volcado de {db_path} a {out_path}...")
    try:
        con = sqlite3.connect(db_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in con.iterdump():
                f.write(f"{line}\n")
        con.close()
        print(f"Volcado completado: {out_path}")
    except Exception as e:
        print(f"Fallo al volcar {db_path}: {e}")
