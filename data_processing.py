# %%

import mysql.connector
import sqlparse
from sqlglot.optimizer import optimize
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot import ParseError, parse_one

# %%

connector = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Zoo"
)

cursor = connector.cursor()

# %%

def provide_schema(page):
  match page:
    case 'SELECT_from_WORLD_Tutorial' | 'SELECT_within_SELECT_Tutorial' | 'SELECT_names':
      return {
        'world': {'name': 'STRING', 'continent': 'STRING', 'area': 'INT', 'population': 'INT', 'gdp': 'INT', 'capital': 'STRING'}
      }
    case 'SELECT_from_Nobel_Tutorial':
      return {
        'nobel': {'yr': 'INT', 'subject': 'STRING', 'winner': 'STRING'}
      }
    case 'The_JOIN_operation':
      return {
        'game': {'id': 'INT', 'mdate': 'STRING', 'stadium': 'STRING', 'team1': 'STRING', 'team2': 'STRING'},
        'goal': {'matchid': 'INT', 'teamid': 'STRING', 'player': 'STRING', 'gtime': 'INT'},
        'eteam': {'id': 'STRING', 'teamname': 'STRING', 'coach': 'STRING'}
      }
    case 'More_JOIN_operations':
      return {
        'movie': {'id': 'INT', 'title': 'STRING', 'yr': 'DECIMAL(4)', 'director': 'INT', 'budget': 'INT', 'gross': 'INT'},
        'actor': {'id': 'INT', 'name': 'STRING'},
        'casting': {'movieid': 'INT', 'actorid': 'INT', 'ord': 'INT'}
      }
    case _:
      return None

def provide_entities(page):
  match page:
    case 'SELECT_from_WORLD_Tutorial' | 'SELECT_within_SELECT_Tutorial' | 'SELECT_names':
      return ['world', 'name', 'continent', 'area', 'population', 'gdp', 'capital']
    case 'SELECT_from_Nobel_Tutorial':
      return ['nobel', 'yr', 'subject', 'winner']
    case 'The_JOIN_operation':
      return ['game', 'id', 'mdate', 'stadium', 'team1', 'team2', 'goal', 'matchid', 'teamid', 'player', 'gtime', 'eteam', 'teamname', 'coach']
    case 'More_JOIN_operations':
      return ['movie', 'id', 'title', 'yr', 'director', 'budget', 'gross', 'actor', 'name', 'casting', 'movieid', 'actorid', 'ord']
    case _:
      return []

# THE KEYWORD 'AS' HAS BEEN TAKEN OUT OF THE FOLLOWING SET OF RESERVED WORDS!
mssql_reserved_words = ['ADD', 'EXTERNAL', 'PROCEDURE', 'ALL', 'FETCH', 'PUBLIC', 'ALTER', 'FILE', 'RAISERROR', 'AND', 'FILLFACTOR', 'READ', 'ANY', 'FOR', 'READTEXT', 'FOREIGN', 'RECONFIGURE', 'ASC', 'FREETEXT', 'REFERENCES', 'AUTHORIZATION', 'FREETEXTTABLE', 'REPLICATION', 'BACKUP', 'FROM', 'RESTORE', 'BEGIN', 'FULL', 'RESTRICT', 'BETWEEN', 'FUNCTION', 'RETURN', 'BREAK', 'GOTO', 'REVERT', 'BROWSE', 'GRANT', 'REVOKE', 'BULK', 'GROUP', 'RIGHT', 'BY', 'HAVING', 'ROLLBACK', 'CASCADE', 'HOLDLOCK', 'ROWCOUNT', 'CASE', 'IDENTITY', 'ROWGUIDCOL', 'CHECK', 'IDENTITY_INSERT', 'RULE', 'CHECKPOINT', 'IDENTITYCOL', 'SAVE', 'CLOSE', 'IF', 'SCHEMA', 'CLUSTERED', 'IN', 'SECURITYAUDIT', 'COALESCE', 'INDEX', 'SELECT', 'COLLATE', 'INNER', 'SEMANTICKEYPHRASETABLE', 'COLUMN', 'INSERT', 'SEMANTICSIMILARITYDETAILSTABLE', 'COMMIT', 'INTERSECT', 'SEMANTICSIMILARITYTABLE', 'COMPUTE', 'INTO', 'SESSION_USER', 'CONSTRAINT', 'IS', 'SET', 'CONTAINS', 'JOIN', 'SETUSER', 'CONTAINSTABLE', 'KEY', 'SHUTDOWN', 'CONTINUE', 'KILL', 'SOME', 'CONVERT', 'LEFT', 'STATISTICS', 'CREATE', 'LIKE', 'SYSTEM_USER', 'CROSS', 'LINENO', 'TABLE', 'CURRENT', 'LOAD', 'TABLESAMPLE', 'CURRENT_DATE', 'MERGE', 'TEXTSIZE', 'CURRENT_TIME', 'NATIONAL', 'THEN', 'CURRENT_TIMESTAMP', 'NOCHECK', 'TO', 'CURRENT_USER', 'NONCLUSTERED', 'TOP', 'CURSOR', 'NOT', 'TRAN', 'DATABASE', 'NULL', 'TRANSACTION', 'DBCC', 'NULLIF', 'TRIGGER', 'DEALLOCATE', 'OF', 'TRUNCATE', 'DECLARE', 'OFF', 'TRY_CONVERT', 'DEFAULT', 'OFFSETS', 'TSEQUAL', 'DELETE', 'ON', 'UNION', 'DENY', 'OPEN', 'UNIQUE', 'DESC', 'OPENDATASOURCE', 'UNPIVOT', 'DISK', 'OPENQUERY', 'UPDATE', 'DISTINCT', 'OPENROWSET', 'UPDATETEXT', 'DISTRIBUTED', 'OPENXML', 'USE', 'DOUBLE', 'OPTION', 'USER', 'DROP', 'OR', 'VALUES', 'DUMP', 'ORDER', 'VARYING', 'ELSE', 'OUTER', 'VIEW', 'END', 'OVER', 'WAITFOR', 'ERRLVL', 'PERCENT', 'WHEN', 'ESCAPE', 'PIVOT', 'WHERE', 'EXCEPT', 'PLAN', 'WHILE', 'EXEC', 'PRECISION', 'WITH', 'EXECUTE', 'PRIMARY', 'WITHIN GROUP', 'EXISTS', 'PRINT', 'WRITETEXT', 'EXIT', 'PROC']

# %%

pages = [
  # ('SELECT_from_WORLD_Tutorial', 8),
  # ('SELECT_from_Nobel_Tutorial', 14),
  # ('The_JOIN_operation', 13),
  # ('SELECT_within_SELECT_Tutorial', 5),
  # ('SELECT_names', 12),
  # ('SELECT_names', 15),
  ('More_JOIN_operations', 12)
]

# %%

for page in pages:
  cursor.execute(f'SELECT txt FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} AND score = 100;')
  raw_queries = cursor.fetchall()
  raw_query_set = { query[0] for query in raw_queries }
  formatted_query_set = { sqlparse.format(query, reindent=True, keyword_case='upper', strip_comments=True) for query in raw_query_set }

  query_asts = set()
  canonicalized_queries = set()
  optimized_queries = set()
  subset_queries = set()
  parse_errors = 0
  other_errors = 0
  schema = provide_schema(page[0])
  keywords_and_entities = mssql_reserved_words + provide_entities(page[0])

  for query in formatted_query_set:

    clean_query = query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '')

    try:
      ast = parse_one(clean_query, dialect='tsql')
      query_asts.add(ast)
      canonicalized_queries.add(canonicalize(ast))
      optimized_queries.add(optimize(ast, schema).sql(dialect='tsql'))
      subset_queries.add(tuple(feature for feature in query.replace('\n', ' ').replace('(', ' ').replace(')', '').replace(',', '').replace('.', ' ').split(' ') if feature in keywords_and_entities))
    except ParseError as err:
      # print('PARSE ERROR:', err)
      # print('FOR QUERY', query)
      parse_errors += 1
      continue
    except BaseException as err:
      # print('OTHER ERROR:', err)
      # print('FOR QUERY:\n', query)
      other_errors += 1
      continue
  
  print(f'RESULTS FOR CATEGORY {page[0]}, QUESTION {page[1]}')
  print(f'TOTAL UNPROCESSED QUERIES: {len(raw_queries)}')
  print(f'TOTAL UNIQUE UNPROCESSED QUERIES: {len(raw_query_set)}')
  print(f'TOTAL UNIQUE FORMATTED QUERIES: {len(formatted_query_set)}')
  print(f'TOTAL UNIQUE ASTs: {len(query_asts)}')
  print(f'TOTAL CANONICALIZED QUERIES: {len(canonicalized_queries)}')
  print(f'TOTAL OPTIMIZED QUERIES: {len(optimized_queries)}')
  print(f'TOTAL SUBSET QUERIES: {len(subset_queries)}')
  print(f'TOTAL PARSE ERRORS: {parse_errors}')
  print(f'TOTAL OTHER ERRORS: {other_errors}')
  
  for q in list(subset_queries)[:10]:
    print(q)

# %%

q = """
SELECT title,
       name
FROM movie
JOIN casting ON (movieid=movie.id
                 AND ord=1)
JOIN actor ON (actorid=actor.id)
WHERE movie.id IN
    (SELECT movieid
     FROM casting
     WHERE actorid IN
         (SELECT id
          FROM actor
          WHERE name='Julie Andrews'))
"""

ast = parse_one(q, dialect='tsql')
print('###################')
print('\n\n\n')
print(repr(ast))
print('\n\n\n')
print('###################')

print('###################')
print('\n\n\n')
print(canonicalize(ast))
print('\n\n\n')
print('###################')

sch = provide_schema('More_JOIN_operations')
print(optimize(ast, sch).sql(dialect='tsql'))
