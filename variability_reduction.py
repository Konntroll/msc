# %%

import mysql.connector
import copy
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
import sqlparse
from sqlparse.tokens import Keyword, Name, Comparison
from sqlglot import ParseError, parse_one

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

def provide_schema(page: str):
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

def provide_entities(page: str):
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

def catch(func, *args, **kwargs):
    global parse_errors, key_errors, other_errors
    try:
      return func(*args, **kwargs)
    except ParseError as err:
      # print(f'PARSE ERROR AT {func.__name__}: {err}')
      parse_errors += 1
      return None
    except KeyError as err:
      # print(f'KEY ERROR AT {func.__name__}: {err}')
      key_errors += 1
      return None
    except BaseException as err:
      # print(f'OTHER ERROR AT {func.__name__}: {err}')
      other_errors += 1
      return None

def find_most_relevant_ast(set_of_asts: set, list_of_asts: list, clean_query_list: list, step: str):  
  print(f'Looking for an AST that accounts for the highest number of original queries at {step}...\n')

  dict_of_results = {}
  
  for query_ast in set_of_asts:
    dict_of_results[query_ast] = [ idx for (idx, listed_ast) in enumerate(list_of_asts) if listed_ast == query_ast ]

  longest_sequence = []
  optimized_query: str

  for key, value in dict_of_results.items():
    if len(value) > len(longest_sequence):
      longest_sequence = value
      optimized_query = key

  print(f'Longest sequence of matches: {len(dict_of_results[optimized_query])}\n')
  print(f'Optimized query: {optimized_query}\n')

  for idx in dict_of_results[optimized_query][:20]:
    print('\n==================================================================\n')
    try:
      print(clean_query_list[idx])
    except IndexError:
      print(f'No query matches {idx}')

# KEYWORDS 'AS' AND 'ASC" HAVE BEEN TAKEN OUT OF THE FOLLOWING SET!
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

# global error counts for use in the optimization process
parse_errors = 0
key_errors = 0
other_errors = 0

# %%

# This section implements several versions of feature subsetting
# and prints out the results for each of the version.

for page in pages:
  cursor.execute(f'SELECT txt FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} AND score = 100;')
  raw_queries = cursor.fetchall()
  raw_query_set = { query[0] for query in raw_queries }
  formatted_query_set = { sqlparse.format(query, reindent=True, keyword_case='upper', strip_comments=True) for query in raw_query_set }
  clean_query_set = { query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '').replace('', '').strip() for query in formatted_query_set }
  features = mssql_keywords + provide_entities(page[0])

  naive_feature_subsets = set()
  for query in clean_query_set:
    naive_feature_subsets.add(tuple(feature for feature in query.replace('\n', ' ').replace('(', ' ').replace(')', '').replace(',', '').replace('.', ' ').split(' ') if feature in features))

  ast_based_feature_subsets_with_aliases = { extract_query_features(query, features) for query in clean_query_set}
  ast_based_feature_subsets_with_numbered_aliases = { extract_query_features(query, features, True, True, True) for query in clean_query_set }
  ast_based_feature_subsets_with_common_alias = { extract_query_features(query, features, True, True) for query in clean_query_set }
  ast_based_feature_subsets_without_aliases = { extract_query_features(query, features, False) for query in clean_query_set }

  print(f'FEATURE SUBSETTING RESULTS FOR {page[0]}, question {page[1]}:')
  print(f'TOTAL NAIVE FEATURE SUBSETS: {len(naive_feature_subsets)}')
  print(f'TOTAL FEATURE SUBSETS WITH ALIASES: {len(ast_based_feature_subsets_with_aliases)}')
  print(f'TOTAL FEATURE SUBSETS WITH NUMBERED STANDARDISED ALIASES: {len(ast_based_feature_subsets_with_numbered_aliases)}')
  print(f'TOTAL FEATURE SUBSETS WITH A COMMON STANDARDISED ALIAS: {len(ast_based_feature_subsets_with_common_alias)}')
  print(f'TOTAL FEATURE SUBSETS WITHOUT ALIASES: {len(ast_based_feature_subsets_without_aliases)}')
  print(f'\n=======================================================\n')


# %%

# This section runs through invidividual optimisation steps
# and prints the results for each step for each question.

# The code necessary to single out the ASTs that account 
# for the highest number of original queries is commented 
# out to expedite processing and declutter the output.
# The code necessary to perform futher reduction of variability
# is commented out as it's not very relevant to the outcome.

for page in pages:
  # resetting error counts
  parse_errors = 0
  key_errors = 0
  other_errors = 0

  # fetching queries
  cursor.execute(f'SELECT txt FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} AND score = 100;')

  # extracting raw queries
  raw_queries = cursor.fetchall()

  # creating a set of raw queries
  raw_query_set = { query[0] for query in raw_queries }

  # creating a set of formatted queries
  formatted_query_set = { sqlparse.format(query, reindent=True, keyword_case='upper', strip_comments=True) for query in raw_query_set }

  # applying a small set of further query cleaning steps
  clean_query_set = { query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '').replace('', '').strip() for query in formatted_query_set }

  # creating ASTs with SQLParse for comparison with SQLGlot
  sqlparse_parsed_asts = { sqlparse.parse(query) for query in clean_query_set }

  # retreiving a schema for the current page - a requirement for many SQLGlot optimisation steps
  schema = provide_schema(page[0])

  # creating ASTs with SQLGlot
  query_ast_set = { parse_one(query, dialect='tsql') for query in clean_query_set if catch(parse_one, query, dialect='tsql') != None }
  # query_ast_list = [ parse_one(query, dialect='tsql') for query in clean_query_set if catch(parse_one, query, dialect='tsql') != None ]

  # The following are SQLGlot optimisation steps taken individually.
  # Normally the optimize method runs all of these as a package and
  # they feed into each other. For the purposes of our study, we ran
  # the steps individually isolating the outputs from each other by
  # creating deep copies of the results obtained at each step.

  ast_set_for_qualification = { copy.deepcopy(ast) for ast in query_ast_set }
  # ast_list_for_qualification = [ copy.deepcopy(ast) for ast in query_ast_list ]

  qualified_queries = { qualify(ast, schema=schema) for ast in ast_set_for_qualification if catch(qualify, ast, schema=schema) != None }
  # qualified_queries_list = [ qualify(ast, schema=schema) for ast in ast_list_for_qualification if catch(qualify, ast, schema=schema) != None ]

  # if page[0] == 'SELECT_from_WORLD_Tutorial':
  #   find_most_relevant_ast(qualified_queries, qualified_queries_list, list(clean_query_set), 'qualification')

  ast_set_for_porjection_pushdown = { copy.deepcopy(ast) for ast in qualified_queries }
  # ast_list_for_porjection_pushdown = [ copy.deepcopy(ast) for ast in qualified_queries_list ]

  with_projections_pushed_down = { pushdown_projections(ast, schema=schema) for ast in ast_set_for_porjection_pushdown }
  # with_projections_pushed_down_list = [ pushdown_projections(ast, schema=schema) for ast in ast_list_for_porjection_pushdown ]

  ast_set_for_query_normalization = { copy.deepcopy(ast) for ast in with_projections_pushed_down }
  # ast_list_for_query_normalization = [ copy.deepcopy(ast) for ast in with_projections_pushed_down_list ]

  normalized_queries = { normalize(ast) for ast in ast_set_for_query_normalization }
  # normalized_queries_list = [ normalize(ast) for ast in ast_list_for_query_normalization ]

  ast_set_for_subquery_unnesting = { copy.deepcopy(ast) for ast in normalized_queries }
  # ast_list_for_subquery_unnesting = [ copy.deepcopy(ast) for ast in normalized_queries_list ]

  with_subqueries_unnested = { unnest_subqueries(ast) for ast in ast_set_for_subquery_unnesting }
  # with_subqueries_unnested_list = [ unnest_subqueries(ast) for ast in ast_list_for_subquery_unnesting ]

  ast_set_for_predicate_pushdown = { copy.deepcopy(ast) for ast in with_subqueries_unnested }
  # ast_list_for_predicate_pushdown = [ copy.deepcopy(ast) for ast in with_subqueries_unnested_list ]

  with_predicates_pushed_down = { pushdown_predicates(ast, dialect='tsql') for ast in ast_set_for_predicate_pushdown if catch(pushdown_predicates, ast, dialect='tsql') != None }
  # with_predicates_pushed_down_list = [ pushdown_predicates(ast, dialect='tsql') for ast in ast_list_for_predicate_pushdown if catch(pushdown_predicates, ast, dialect='tsql') != None ]

  # if page[0] == 'SELECT_from_WORLD_Tutorial':
  #   find_most_relevant_ast(with_predicates_pushed_down, with_predicates_pushed_down_list, list(clean_query_set), 'predicate pushdown')

  ast_set_for_join_optimization = { copy.deepcopy(ast) for ast in with_predicates_pushed_down }
  with_joins_optimized = { optimize_joins(ast) for ast in ast_set_for_join_optimization }

  ast_set_for_subquery_elimination = { copy.deepcopy(ast) for ast in with_joins_optimized }
  with_subqueries_eliminted = { eliminate_subqueries(ast) for ast in ast_set_for_subquery_elimination }

  ast_set_for_merging_of_subqueries = { copy.deepcopy(ast) for ast in with_subqueries_eliminted }
  with_subqueries_merged = { merge_subqueries(ast) for ast in ast_set_for_merging_of_subqueries if catch(merge_subqueries, ast) != None }

  ast_set_for_join_elimination = { copy.deepcopy(ast) for ast in with_subqueries_merged }
  with_joins_eliminated = { eliminate_joins(ast) for ast in ast_set_for_join_elimination }

  ast_set_for_cte_elimination = { copy.deepcopy(ast) for ast in with_joins_eliminated }
  with_ctes_eliminated = { eliminate_ctes(ast) for ast in ast_set_for_cte_elimination }

  ast_set_for_application_of_quote_identifiers = { copy.deepcopy(ast) for ast in with_ctes_eliminated }
  with_quote_identifiers = { quote_identifiers(ast, dialect='tsql') for ast in ast_set_for_application_of_quote_identifiers }

  ast_set_for_annotation_of_types = { copy.deepcopy(ast) for ast in with_quote_identifiers }
  with_types_annotated = { annotate_types(ast, schema=schema) for ast in ast_set_for_annotation_of_types }

  ast_set_for_canonicalization = { copy.deepcopy(ast) for ast in with_types_annotated }
  canonicalized_queries = { canonicalize(ast) for ast in ast_set_for_canonicalization }

  ast_set_for_simplification = { copy.deepcopy(ast) for ast in canonicalized_queries }
  simplified_queries = { simplify(ast, dialect='tsql') for ast in ast_set_for_simplification }

  ast_set_for_assisted_optimization = { copy.deepcopy(ast) for ast in simplified_queries }
  optimized_based_on_simplified = { optimize(ast, schema=schema, dialect='tsql') for ast in ast_set_for_assisted_optimization if catch(optimize, ast, schema=schema) != None }

  print(f'\nRESULTS FOR CATEGORY {page[0]}, QUESTION {page[1]}')
  print(f'DISTINCT CLEAN QUERIES: {len(clean_query_set)}')
  print(f'SQLParse ASTs: {len(sqlparse_parsed_asts)}')
  print(f'SQLGlot ASTs: {len(query_ast_set)}')
  print(f'QUALIFIED: {len(qualified_queries)}')
  print(f'WITH PROJECTION PUSHDOWN: {len(with_projections_pushed_down)}')
  print(f'NORMALIZED: {len(normalized_queries)}')
  print(f'WITH SUBQUERIES UNNESTED: {len(with_subqueries_unnested)}')
  print(f'WITH PREDICATE PUSHDOWN: {len(with_predicates_pushed_down)}')
  print(f'WITH JOINS OPTIMIZED: {len(with_joins_optimized)}')
  print(f'WITH SUBQUERIES ELIMINATED: {len(with_subqueries_eliminted)}')
  print(f'WITH SUBQUERIES MERGED: {len(with_subqueries_merged)}')
  print(f'WITH JOINS ELIMINATED: {len(with_joins_eliminated)}')
  print(f'WITH CTEs ELIMINATED: {len(with_ctes_eliminated)}')
  print(f'WITH QUOTE IDENTIFIERS APPLIED: {len(with_quote_identifiers)}')
  print(f'WITH TYPES ANNOTATED: {len(with_types_annotated)}')
  print(f'CANONICALIZED: {len(canonicalized_queries)}')
  print(f'SIMPLIFIED: {len(simplified_queries)}')
  print(f'\nOPTIMIZED: {len(optimized_based_on_simplified)}')

  # reduced = { query.__str__() for query in optimized_based_on_simplified }

  # print(f'\nREDUCED: {len(reduced)}')

  # reparsed = { parse_one(query, dialect='tsql') for query in reduced if catch(parse_one, query, dialect='tsql') != None}
  # reoptimized = { optimize(ast, schema=schema, dialect='tsql') for ast in reparsed if catch(optimize, ast, schema=schema) != None }

  # print(f'RE-OPTIMIZED: {len(reoptimized)}')

  print(f'\nPARSE ERRORS: {parse_errors}')
  print(f'KEY ERRORS: {key_errors}')
  print(f'OTHER ERRORS: {other_errors}')
  print('\n==================================================================')


# %%
