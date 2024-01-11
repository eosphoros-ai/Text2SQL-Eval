import re
import sqlite3
from sqlgpt_parser.parser.mysql_parser import parser
from sqlgpt_parser.parser.parser_utils import ParserUtils


def find_column_in_tables(column, tables):
    if len(column) == 1 or isinstance(column[0], dict) or isinstance(column[0], list):
        in_tables = None
        if len(column) == 1:
            column_name = column[0]
        else:
            column_name = column[1]
            if isinstance(column[0], dict):
                in_tables = column[0]
            if isinstance(column[0], list):
                in_tables = {x: x for x in column[0]}

        table = find_column_name_in_tables(tables, column_name, in_tables)
    else:
        in_tables, column_name = column
        if column_name == "*":
            table = [in_tables]
        else:
            in_tables = {in_tables: in_tables}
            table = find_column_name_in_tables(tables, column_name, in_tables)
    return column_name, table


def find_column_name_in_tables(tables, column_name, in_tables=None):
    for _, value in in_tables.items():
        if isinstance(value, dict):
            return []
    table_result = []
    for table, table_column_names in tables.items():
        if column_name.lower() in table_column_names:
            if in_tables is not None:
                for alias, table_name in in_tables.items():
                    if table_name.lower() == table:
                        table_result.append(table)
            else:
                table_result.append(table)
    return table_result


def isConstCanFind(sql, db_list, tables):
    statement = ParserUtils.format_statement(parser.parse(sql))
    compare_filters = statement.compare_filters
    already_used_compare_filters = set()
    temp_compare_filters = []
    for used_table_column, compare_type, compare_value in compare_filters:
        column_name, table_names = find_column_in_tables(used_table_column, tables)
        for table_name in table_names:
            temp_compare_filters.append([[table_name, column_name], compare_type, compare_value])
    compare_filters = temp_compare_filters
    temp_compare_filters = []
    for compare_filter in compare_filters:
        if compare_filter[1] != '=':
            continue
        compare_filter_str = f"{compare_filter[0][0]}.{compare_filter[0][1]}{compare_filter[1]}{compare_filter[2]}"
        if compare_filter_str not in already_used_compare_filters:
            temp_compare_filters.append(compare_filter)
            already_used_compare_filters.add(compare_filter_str)
    compare_filters = temp_compare_filters
    may_be_other_field = []
    for compare_filter in compare_filters:
        used_table_column, compare_type, compare_value = compare_filter
        table, column = used_table_column
        sql = "SELECT {}.{} FROM {} WHERE {}.{} {} '{}'". \
            format(table, column, table, table, column, compare_type, str(compare_value))
        is_empty = True
        for db in db_list:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()
            cursor.execute(sql)
            if len(cursor.fetchall()) != 0:
                is_empty = False
                break
        if not is_empty:
            continue
        other_field = {
            "table": table,
            "not_right_column": column,
            "compare_type": compare_type,
            "compare_value": compare_value
        }
        may_in_columns = []
        table_columns = tables[table]
        for table_column in table_columns:
            sql = "SELECT {}.{} FROM {} WHERE {}.{} {} '{}'". \
                format(table, table_column, table, table, table_column, compare_type, str(compare_value))
            for db in db_list:
                conn = sqlite3.connect(db)
                cursor = conn.cursor()
                cursor.execute(sql)
                if len(cursor.fetchall()) != 0:
                    may_in_columns.append(table_column)
        other_field["may_in_columns"] = may_in_columns
        may_be_other_field.append(other_field)
    return may_be_other_field


def convert_schema(prompt):
    tables = {}
    foreign_keys = []
    for table_column in prompt.split("|")[1:]:
        if table_column.find(":") == -1:
            foreign_keys.append(table_column.strip())
            continue
        table_column = re.sub(r'\(.*?\)', '', table_column)
        table, column_str = table_column.split(":")
        table = table.strip()
        columns = []
        for column in column_str.split(","):
            if "(" in column:
                column = column.split("(")[0]
            try:
                column = column.split(".")[1].strip()
            except:
                continue
            columns.append(column)
        tables[table] = columns
    return tables, foreign_keys


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def fetch_column_all_value(column, db_list, table):
    results = set()
    for db in db_list:
        connection = sqlite3.connect(db)
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL;")
            for item in cursor.fetchall():
                results.add(item[0])
        except:
            continue
    return results


def convert_tables_to_create_table(tables, primary_keys, foreign_keys):
    create_db = ""
    for table, columns in tables.items():
        create_table = f"CREATE TABLE {table} (\n"
        for column in columns:
            create_table += f"{column},\n"
        if table in primary_keys and primary_keys[table][0] in columns:
            create_table += f"primary key ({primary_keys[table][0]}\n"
        for foreign_key in foreign_keys:
            primary_table, primary_key, foreign_table, foreign_key = (
                get_foreign_key_primary_key(foreign_key, primary_keys))
            create_table += f"foreign key({primary_key.strip()}) references {foreign_table}({foreign_key.strip()}),\n"
        create_table = create_table.strip()[:-1] + '\n'
        create_table += ");\n"
        create_db += create_table
    return create_db


def get_foreign_key_primary_key(foreign_key, primary_keys):
    left, right = foreign_key.split("=")
    left_table, left_column = left.split(".")
    right_table, right_column = right.split(".")
    if left_table in primary_keys and left_column == primary_keys[left_table][0]:
        return left_table, left_column, right_table, right_column
    return right_table, right_column, left_table, left_column


def convert_meta_data(tables_json):
    meta_data = {}
    for db in tables_json:
        db_id = db['db_id']
        table_names = db["table_names_original"]
        tables_column = []
        tables_column_with_type = []
        for i in range(len(table_names)):
            tables_column.append([])
            tables_column_with_type.append([])
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        for column, column_type in zip(column_names, column_types):
            if column[0] == -1:
                continue
            tables_column[column[0]].append(column[1].lower())
            tables_column_with_type[column[0]].append([column[1].lower(), column_type])
        table_name_to_column_dicts = {}
        table_name_to_column_with_type_dicts = {}
        for index in range(len(tables_column)):
            table_name_to_column_dicts[table_names[index].lower()] = tables_column[index]
        for index in range(len(tables_column_with_type)):
            table_name_to_column_with_type_dicts[table_names[index].lower()] = tables_column_with_type[index]
        foreign_keys = {}
        for foreign_key in db["foreign_keys"]:
            key1, key2 = foreign_key
            table_name1, column_name1 = table_names[column_names[key1][0]].lower(), column_names[key1][1].lower()
            table_name2, column_name2 = table_names[column_names[key2][0]].lower(), column_names[key2][1].lower()
            if table_name1 not in foreign_keys:
                foreign_keys[table_name1] = {}
            table_foreign_keys = foreign_keys[table_name1]
            if column_name1 not in table_foreign_keys:
                table_foreign_keys[column_name1] = []
            table_foreign_keys[column_name1].append([table_name2, column_name2])
            # foreign_keys.append({table_name1: column_name1, table_name2: column_name2})

        primary_keys = {}
        for primary_key in db["primary_keys"]:
            column_name_primary_keys = []
            if isinstance(primary_key, int):
                table_name = table_names[column_names[primary_key][0]].lower()
                column_name_primary_keys.append(column_names[primary_key][1].lower())
            else:
                for primary_key_index in primary_key:
                    table_name = table_names[column_names[primary_key_index][0]].lower()
                    column_name_primary_keys.append(column_names[primary_key_index][1].lower())
            primary_keys[table_name] = column_name_primary_keys

        meta_data[db_id] = {"tables": table_name_to_column_dicts, "foreign_keys": foreign_keys,
                            "primary_keys": primary_keys, 'tables_column_type': table_name_to_column_with_type_dicts}

    return meta_data
