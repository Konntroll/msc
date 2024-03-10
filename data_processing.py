# %%

import mysql.connector
import sqlparse
from sqlglot.optimizer import optimize
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.unnest_subqueries import unnest_subqueries
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.qualify_columns import quote_identifiers
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.simplify import simplify
from sqlparse.tokens import Keyword, Name, Comparison
from sqlglot import Expression, ParseError, parse_one, exp

connector = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Zoo"
)

cursor = connector.cursor()

def extract_query_features(query: str, features: list[str], include_aliases: bool = True, standardise_aliases: bool = False, numbered_aliases: bool = False):

  subset = []
  aliases = set()

  for token in sqlparse.parse(query)[0].flatten():
    if token.ttype is Name:
      if token.value in features:
        subset.append(token.value)
      if token.value not in features and include_aliases:
        subset.append(token.value)
        if (standardise_aliases):
          aliases.add(token.value)
    if token.ttype is Keyword and token.value not in ('AS', 'ASC'):
      subset.append(token.value)
    if token.ttype is Comparison and token.value == 'LIKE':
      subset.append(token.value)

  if (include_aliases and standardise_aliases):
    if (numbered_aliases):
      for idx, alias in enumerate(aliases):
        for fIdx, feature in enumerate(subset):
          if feature == alias:
            subset[fIdx] = f'A{idx + 1}'

    if (not numbered_aliases):
      for alias in aliases:
        for fIdx, feature in enumerate(subset):
          if feature == alias:
            subset[fIdx] = 'entity_alias'
  
  return tuple(subset)

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
      return []

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

# THE KEYWORDS 'AS' AND 'ASC" HAVE BEEN TAKEN OUT OF THE FOLLOWING SET!
mssql_keywords = ['CURRENT_TIMEZONE', 'CURRENT_TIMEZONE_ID', 'DATE_BUCKET', 'DIFFERENCE', 'FORMAT', 'QUOTENAME', 'REPLICATE', 'REVERSE', 'UNICODE', 'SOUNDEX', 'STRING_AGG', 'STRING_ESCAPE', 'SUBSTRING', 'TRANSLATE', 'RAND', 'ISNULL', 'ISNUMERIC', 'LAG', 'LEAD', 'USER_NAME', 'SESSIONPROPERTY', 'CAST', 'CONVERT', 'TRY_CAST', 'TRY_CONVERT', 'PARSE', 'TRY_PARSE', 'ROUND', 'SIGN', 'SUM', 'DATEADD', 'DATEDIFF', 'DATEDIFF_BIG', 'DATENAME', 'DATEPART', 'DATETIMEFROMPARTS', 'DATETIME2FROMPARTS', 'DATETIMEOFFSETFROMPARTS', 'DATETRUNC', 'DAY', 'EOMONTH', 'GETDATE', 'ISDATE', 'SMALLDATETIMEFROMPARTS', 'SWITCHOFFSET', 'SYSUTCDATETIME', 'TIMEFROMPARTS', 'TODATETIMEOFFSET', '', 'GETUTCDATE', 'MONTH', 'YEAR', 'FLOOR', 'MAX', 'MIN', 'ABS', 'AVG', 'CEILING', 'UPPER', 'SUBSTR', 'STUFF', 'STR', 'SPACE', 'PATINDEX', 'REPLACE', 'NCHAR', 'TRIM', 'LTRIM', 'RTRIM', 'LOWER', 'LEN', 'DATALENGTH', 'CHAR', 'CHARINDEX', 'CONCAT', 'CONCAT_WS', 'ADD', 'EXTERNAL', 'PROCEDURE', 'ALL', 'COUNT', 'FETCH', 'PUBLIC', 'ALTER', 'FILE', 'RAISERROR', 'AND', 'FILLFACTOR', 'READ', 'ANY', 'FOR', 'READTEXT', 'FOREIGN', 'RECONFIGURE', 'FREETEXT', 'REFERENCES', 'AUTHORIZATION', 'FREETEXTTABLE', 'REPLICATION', 'BACKUP', 'FROM', 'RESTORE', 'BEGIN', 'FULL', 'RESTRICT', 'BETWEEN', 'FUNCTION', 'RETURN', 'BREAK', 'GOTO', 'REVERT', 'BROWSE', 'GRANT', 'REVOKE', 'BULK', 'GROUP', 'RIGHT', 'BY', 'HAVING', 'ROLLBACK', 'CASCADE', 'HOLDLOCK', 'ROWCOUNT', 'CASE', 'IDENTITY', 'ROWGUIDCOL', 'CHECK', 'IDENTITY_INSERT', 'RULE', 'CHECKPOINT', 'IDENTITYCOL', 'SAVE', 'CLOSE', 'IF', 'SCHEMA', 'CLUSTERED', 'IN', 'SECURITYAUDIT', 'COALESCE', 'INDEX', 'SELECT', 'COLLATE', 'INNER', 'SEMANTICKEYPHRASETABLE', 'COLUMN', 'INSERT', 'SEMANTICSIMILARITYDETAILSTABLE', 'COMMIT', 'INTERSECT', 'SEMANTICSIMILARITYTABLE', 'COMPUTE', 'INTO', 'SESSION_USER', 'CONSTRAINT', 'IS', 'SET', 'CONTAINS', 'JOIN', 'SETUSER', 'CONTAINSTABLE', 'KEY', 'SHUTDOWN', 'CONTINUE', 'KILL', 'SOME', 'CONVERT', 'LEFT', 'STATISTICS', 'CREATE', 'LIKE', 'SYSTEM_USER', 'CROSS', 'LINENO', 'TABLE', 'CURRENT', 'LOAD', 'TABLESAMPLE', 'CURRENT_DATE', 'MERGE', 'TEXTSIZE', 'CURRENT_TIME', 'NATIONAL', 'THEN', 'CURRENT_TIMESTAMP', 'NOCHECK', 'TO', 'CURRENT_USER', 'NONCLUSTERED', 'TOP', 'CURSOR', 'NOT', 'TRAN', 'DATABASE', 'NULL', 'TRANSACTION', 'DBCC', 'NULLIF', 'TRIGGER', 'DEALLOCATE', 'OF', 'TRUNCATE', 'DECLARE', 'OFF', 'TRY_CONVERT', 'DEFAULT', 'OFFSETS', 'TSEQUAL', 'DELETE', 'ON', 'UNION', 'DENY', 'OPEN', 'UNIQUE', 'DESC', 'OPENDATASOURCE', 'UNPIVOT', 'DISK', 'OPENQUERY', 'UPDATE', 'DISTINCT', 'OPENROWSET', 'UPDATETEXT', 'DISTRIBUTED', 'OPENXML', 'USE', 'DOUBLE', 'OPTION', 'USER', 'DROP', 'OR', 'VALUES', 'DUMP', 'ORDER', 'VARYING', 'ELSE', 'OUTER', 'VIEW', 'END', 'OVER', 'WAITFOR', 'ERRLVL', 'PERCENT', 'WHEN', 'ESCAPE', 'PIVOT', 'WHERE', 'EXCEPT', 'PLAN', 'WHILE', 'EXEC', 'PRECISION', 'WITH', 'EXECUTE', 'PRIMARY', 'WITHIN GROUP', 'EXISTS', 'PRINT', 'WRITETEXT', 'EXIT', 'PROC']

pages = [
  ('SELECT_from_WORLD_Tutorial', 8),
  ('SELECT_from_Nobel_Tutorial', 14),
  ('The_JOIN_operation', 13),
  ('SELECT_within_SELECT_Tutorial', 5),
  ('SELECT_names', 12),
  ('SELECT_names', 15),
  ('More_JOIN_operations', 12)
]


# %%

for page in pages:
  cursor.execute(f'SELECT txt FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} AND score = 100;')
  raw_queries = cursor.fetchall()
  raw_query_set = { query[0] for query in raw_queries }
  formatted_query_set = { sqlparse.format(query, reindent=True, keyword_case='upper', strip_comments=True) for query in raw_query_set }
  clean_query_set = { query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '').replace('', '') for query in formatted_query_set }
  sqlparse_parsed_asts = { sqlparse.parse(query) for query in clean_query_set }

  query_asts = set()
  qualified_queries = set()
  with_projections_pushed_down = set()
  normalized_queries = set()
  with_subqueries_unnested = set()
  with_predicates_pushed_down = set()
  with_joins_optimized = set()
  with_subqueries_eliminted = set()
  with_subqueries_merged = set()
  with_joins_eliminated = set()
  with_ctes_eliminated = set()
  with_quote_identifiers = set()
  with_types_annotated = set()
  canonicalized_queries = set()
  simplified_queries = set()
  optimized_queries = set()
  subset_queries = set()
  ast_based_feature_subsets_with_aliases = set()
  ast_based_feature_subsets_with_numbered_aliases = set()
  ast_based_feature_subsets_with_common_alias = set()
  ast_based_feature_subsets_without_aliases = set()
  parse_errors = 0
  other_errors = 0
  schema = provide_schema(page[0])
  features = mssql_keywords + provide_entities(page[0])

  for query in clean_query_set:

    try:
      ast = parse_one(query, dialect='tsql')
      query_asts.add(ast)
      qualified_queries.add(qualify(ast, schema=schema))
      with_projections_pushed_down.add(pushdown_projections(ast, schema=schema))
      normalized_queries.add(normalize(ast))
      with_subqueries_unnested.add(unnest_subqueries(ast))
      with_predicates_pushed_down.add(pushdown_predicates(ast, dialect='tsql'))
      with_joins_optimized.add(optimize_joins(ast))
      with_subqueries_eliminted.add(eliminate_subqueries(ast))
      with_subqueries_merged.add(merge_subqueries(ast))
      with_joins_eliminated.add(eliminate_joins(ast))
      with_ctes_eliminated.add(eliminate_ctes(ast))
      with_quote_identifiers.add(quote_identifiers(ast, dialect='tsql'))
      with_types_annotated.add(annotate_types(ast, schema=schema))
      canonicalized_queries.add(canonicalize(ast))
      simplified_queries.add(simplify(ast, dialect='tsql'))
      optimized_queries.add(optimize(ast, schema).sql(dialect='tsql'))
      subset_queries.add(tuple(feature for feature in query.replace('\n', ' ').replace('(', ' ').replace(')', '').replace(',', '').replace('.', ' ').split(' ') if feature in features))
      ast_based_feature_subsets_with_aliases.add(extract_query_features(query, features))
      ast_based_feature_subsets_with_numbered_aliases.add(extract_query_features(query, features, True, True, True))
      ast_based_feature_subsets_with_common_alias.add(extract_query_features(query, features, True, True))
      ast_based_feature_subsets_without_aliases.add(extract_query_features(query, features, False))
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
  print(f'TOTAL UNIQUE CLEAN QUERIES: {len(clean_query_set)}')
  print(f'TOTAL UNIQUE ASTs: {len(query_asts)}')
  print(f'TOTAL SQLPARSE ASTs: {len(sqlparse_parsed_asts)}')
  print(f'TOTAL QUALIFIED QUERIES: {len(qualified_queries)}')
  print(f'TOTAL WITH PROJECTIONS PUSHED DOWN: {len(with_projections_pushed_down)}')
  print(f'TOTAL NORMALIZED QUERIES: {len(normalized_queries)}')
  print(f'TOTAL WITH SUBQUERIES UNNESTED: {len(with_subqueries_unnested)}')
  print(f'TOTAL WITH PREDICATES PUSHED DOWN: {len(with_predicates_pushed_down)}')
  print(f'TOTAL WITH JOINS OPTIMIZED: {len(with_joins_optimized)}')
  print(f'TOTAL WITH SUBQUERIES ELIMINATED: {len(with_subqueries_eliminted)}')
  print(f'TOTAL WITH SUBQUERIES MERGED: {len(with_subqueries_merged)}')
  print(f'TOTAL WITH JOINS ELIMINATED: {len(with_joins_eliminated)}')
  print(f'TOTAL WITH CTEs ELIMINATED: {len(with_ctes_eliminated)}')
  print(f'TOTAL WITH QUOTES IDENTIFIED: {len(with_quote_identifiers)}')
  print(f'TOTAL WITH TYPES ANNOTATED: {len(with_types_annotated)}')
  print(f'TOTAL CANONICALIZED QUERIES: {len(canonicalized_queries)}')
  print(f'TOTAL SIMPLIFIED QUERIES: {len(simplified_queries)}')
  print(f'TOTAL OPTIMIZED QUERIES: {len(optimized_queries)}')
  print(f'TOTAL IN NAIVE QUERY SUBSET: {len(subset_queries)}')
  print(f'TOTAL FEATURE SUBSETS WITH ALIASES: {len(ast_based_feature_subsets_with_aliases)}')
  print(f'TOTAL FEATURE SUBSETS WITH NUMBERED STANDARDISED ALIASES: {len(ast_based_feature_subsets_with_numbered_aliases)}')
  print(f'TOTAL FEATURE SUBSETS WITH A COMMON STANDARDISED ALIAS: {len(ast_based_feature_subsets_with_common_alias)}')
  print(f'TOTAL FEATURE SUBSETS WITHOUT ALIASES: {len(ast_based_feature_subsets_without_aliases)}')
  print(f'TOTAL PARSE ERRORS: {parse_errors}')
  print(f'TOTAL OTHER ERRORS: {other_errors}')
  
