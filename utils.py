from sqlglot import ParseError
from globals import errors
import sqlparse
from sqlparse.tokens import Keyword, Name, Comparison, DML

def catch(func, *args, **kwargs):
    try:
      return func(*args, **kwargs)
    except ParseError as err:
      # print(f'PARSE ERROR AT {func.__name__}: {err}')
      errors.parse_errors += 1
      return None
    except KeyError as err:
      # print(f'KEY ERROR AT {func.__name__}: {err}')
      errors.key_errors += 1
      return None
    except BaseException as err:
      # print(f'OTHER ERROR AT {func.__name__}: {err}')
      errors.other_errors += 1
      return None

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

def extract_query_features(query: str, features: list[str], include_aliases: bool = True, standardise_aliases: bool = False, numbered_aliases: bool = False):
    
  subset = []
  aliases = set()

  try:
    for token in sqlparse.parse(query)[0].flatten():
      if token.ttype is Name:
        if token.value in features:
          try:
            subset.append(token.value)
          except:
            print('ERRORED INSIDE ON QUERY:', query)
        if token.value not in features and include_aliases:
          subset.append(token.value)
          if (standardise_aliases):
            aliases.add(token.value)
      if (token.ttype is Keyword or token.ttype is DML) and token.value not in ('AS', 'ASC'):
        subset.append(token.value)
      if token.ttype is Comparison and token.value == 'LIKE':
        subset.append(token.value)
  except IndexError:
    print('ERRORED ON QUERY:', query)

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

def cleanup(query: str) -> str:
  formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper', strip_comments=True)
  return formatted_query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '').replace('', '').strip()