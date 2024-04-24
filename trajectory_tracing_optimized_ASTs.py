# %%

import mysql.connector
import sqlparse
import copy
from sqlglot.optimizer import optimize
from sqlglot import Expression, ParseError, parse_one, diff
from globals import pages, optimized_asts_for_passing_queries, Errors
from utils import provide_schema

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import math

# %%

connector = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Zoo"
)

cursor = connector.cursor()

# %%

errors = Errors(0, 0, 0)

class AttemptRecord:
  id: int
  username: str
  query_txt: str
  query_exp: Expression | None
  score: int | None
  is_sound: bool

  def __init__(self, id: int, username: str, txt: str, exp: Expression | None, score: int | None, errno: int | None):
    self.id = id
    self.username = username
    self.query_txt = txt
    self.query_exp = exp
    self.score = score
    self.is_sound = errno is None
  
  def __str__(self):
    return f'''
                ID: {self.id}
                USER NAME: {self.username}
                SCORE: {self.score}
                TXT LEN: {len(self.query_txt)}
                EXP TYPE: {type(self.query_exp)}
                SOUND: {self.is_sound}
            '''

# Attempt to optimize the supplied query or log an error if encountered.

def provide_optimized_expression(page: tuple, query: str, errno: int | None):
  if errno is not None:
    return None
  schema = provide_schema(page[0])
  formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper', strip_comments=True)
  clean_query = formatted_query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '').replace('', '').strip()
  try:
    ast = parse_one(clean_query, dialect='tsql')
    return optimize(ast, schema, dialect='tsql')
  except ParseError as err:
    # print(f'PARSE ERROR: {err}')
    errors.parse_errors += 1
    return None
  except KeyError as err:
    # print(f'PARSE ERROR: {err}')
    errors.key_errors += 1
    return None
  except BaseException as err:
    # print(f'OTHER ERROR: {err}')
    errors.other_errors += 1
    return None

# Count the number of edits in the supplied edit script. Keeps are disregarded.

def get_edit_distance(diff: list):
  removes = 0
  keeps = 0
  inserts = 0
  moves = 0
  updates = 0

  for edit in diff:
    if str(edit).startswith('Remove'):
      removes += 1
    if str(edit).startswith('Keep'):
      keeps += 1
    if str(edit).startswith('Insert'):
      inserts += 1
    if str(edit).startswith('Move'):
      moves += 1
    if str(edit).startswith('Update'):
      updates += 1
  
  total = removes + inserts + moves + updates

  return total

# Find a solution that is most similar to the supplied attempt.
# This step is very slow as implemented here. Took more than a
# to run on a regular no-frills laptop. May need rethinkging.

def find_most_similar_passing_query(page, attempt: Expression):
  passing_queries = optimized_asts_for_passing_queries[page]
  
  least_distance = 999999999
  reference = Expression()
  for q in passing_queries:
    curr_distance = get_edit_distance(diff(q, attempt))
    if curr_distance < least_distance:
      least_distance = curr_distance
      reference = q
      if least_distance == 1:
        break
  return (reference)


# %%

# Retrieve user names

users = {}

for page in pages:
  cursor.execute(f'SELECT DISTINCT(wgUserName) FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} ORDER BY wgUserName;')

  user_names = [ user_name[0] for user_name in cursor.fetchall() ]

  users[page] = user_names


# %%

# Log errors encountered during processing of attempts

errors_by_page = {}
attempts = {}
for page in pages:
  errors = Errors(0, 0, 0)
  print(f'Fetching for {page[0]}, {page[1]}...')
  user_names = users[page]

  cursor.execute(f'SELECT id, wgUserName, txt, score, errno FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} AND wgUserName IN (\'{"\', \'".join(user_names)}\') ORDER BY wgUserName, id')
  response = cursor.fetchall()
  attempts[page] = [ AttemptRecord(record[0], record[1], record[2], provide_optimized_expression(page, record[2], record[4]), record[3], record[4]) for record in response ]
  errors_by_page[page] = copy.deepcopy(errors)
  print(errors_by_page[page])
  print(attempts[page][0])



# %%

# Store reference solution for each user
# This is either the user's own solution if they reached it,
# or a solution most similar to their last attempt.

ref_solutions_by_page_by_user = {}

for page in pages:
  ref_solutions_by_page_by_user[page] = {}
  for username in users[page]:
    user_attempts: list[AttemptRecord] = [ attempt for attempt in attempts[page] if attempt.username == username ]
    sound_attempts = [ record for record in user_attempts if record.is_sound ]
    
    if len(sound_attempts) == 0:
      ref_solutions_by_page_by_user[page][username] = None
      continue
    
    try:
      last_valid_attempt = [ record for record in sound_attempts if record.query_exp is not None ][-1]
    except IndexError:
      ref_solutions_by_page_by_user[page][username] = None
      continue

    if last_valid_attempt.score == 100:
      ref_solutions_by_page_by_user[page][username] = last_valid_attempt.query_exp
    else:
      ref_exp = find_most_similar_passing_query(page, last_valid_attempt.query_exp)
      ref_solutions_by_page_by_user[page][username] = ref_exp


# %%

for page in pages:
  page_errors: Errors = errors_by_page[page]
  print(f'Errors for {page[0]}, {page[1]}:')
  print(f'Attempts for page: {len(attempts[page])}')
  print(f'Parse errors: {page_errors.parse_errors}')
  print(f'Key errors: {page_errors.key_errors}')
  print(f'Other errors: {page_errors.other_errors}')

# %%

# Store reference solutions in database

sql = 'INSERT INTO ref_solutions (pg, qnum, username, ref_solution) VALUES (%s, %s, %s, %s)'
for page in pages:
  insert: list[tuple] = []
  print(f'Processing page {page[0]}...')
  for username in users[page]:
    insert.append((page[0], page[1], username, str(ref_solutions_by_page_by_user[page][username])))
  
  btm = 0
  top = 1000
  while btm < len(insert):
    cursor.executemany(sql, insert[btm:top])
    connector.commit()
    btm += 1000
    top += 1000
    print(f'Inserted {cursor.rowcount} rows.')

# %%

# Generate trajectories

trajectories = {}

for page in pages:
  print(f'Tracing trajectories for {page[0]}, question {page[1]}...')
  trajectories[page] = {}
  for username in users[page]:
    reference = ref_solutions_by_page_by_user[page][username]
    if reference is None:
      trajectories[page][username] = None
      continue
    records: list[AttemptRecord] = [ attempt for attempt in attempts[page] if attempt.username == username and attempt.is_sound ]
    distances = [ get_edit_distance(diff(record.query_exp, reference)) for record in records if record.query_exp is not None ]
    trajectories[page][username] = distances

# %%

# Store obtained trajectories in database

sql = 'INSERT INTO trajectories (pg, qnum, username, trajectory) VALUES (%s, %s, %s, %s)'
for page in pages:
  insert: list[tuple] = []
  print(f'Processing page {page[0]}...')
  for username in users[page]:
    trajectory = trajectories[page][username]
    if trajectory is not None:
      trajectory = ', '.join([str(dist) for dist in trajectory])
    insert.append((page[0], page[1], username, trajectory))

  cursor.executemany(sql, insert)

  connector.commit()

  print(f'Inserted {cursor.rowcount} rows.')

# %%

# Retreive trajectories from DB

trajectories = {}

for page in pages:
  print(f'Processing {page[0]}...')
  trajectories[page] = {}
  for username in users[page]:
    cursor.execute(f'SELECT trajectory FROM trajectories WHERE pg = "{page[0]}" AND qnum = {page[1]} AND username = "{username}"')
    seq = cursor.fetchall()[0][0]
    if seq is not None:
      trajectories[page][username] = [ int(item) for item in seq.split(', ') ]
    else:
      trajectories[page][username] = None

# %%

# Produce bar charts of trajectory distributions by length by question

lengths = {}
for page in pages:
  lengths[page] = {}
  for idx in range(1, 21):
    lengths[page][f'Solved in {idx}'] = 0
    lengths[page][f'Unsolved in {idx}'] = 0
  for username in users[page]:
    trajectory = trajectories[page][username]
    if trajectory is not None and len(trajectory) < 21:
      if (trajectory[-1] == 0):
        lengths[page][f'Solved in {len(trajectory)}'] += 1
      else:
        lengths[page][f'Unsolved in {len(trajectory)}'] += 1

for page in pages:
  solved_keys = [ int(key.replace('Solved in ', '')) for key in lengths[page].keys() if key.startswith('Solved in ')]
  solved_values = [ value for key, value in lengths[page].items() if key.startswith('Solved in ') ]
  unsolved_keys = [ int(key.replace('Unsolved in ', '')) for key in lengths[page].keys() if key.startswith('Unsolved in ')]
  unsolved_values = [ value for key, value in lengths[page].items() if key.startswith('Unsolved in ') ]

  bar_width = .4

  solved_positions = np.arange(len(solved_keys))
  unsolved_positions = solved_positions + bar_width

  plt.figure(figsize=(12, 6))
  plt.bar(solved_positions, solved_values, width=bar_width, color='orange', label='Solved')
  plt.bar(unsolved_positions, unsolved_values, width=bar_width, color='blue', label='Unsolved')
  plt.xticks(solved_positions + bar_width / 2, solved_keys)
  plt.xlabel('Trajectory length')
  plt.ylabel('Number of trajectories of given length')
  solved_patch = mpatches.Patch(color='orange', label='Solved')
  unsolved_patch = mpatches.Patch(color='blue', label='Unsolved')
  plt.legend(handles=[solved_patch, unsolved_patch])
  plt.title(f'{' '.join(page[0].split('_'))}, question {page[1]}')
  plt.show()

# %%

# Generate intertia elbow plots for trajectories of given lengths and result

permutations = [
  (5, True),
  (5, False),
  (10, True),
  (10, False),
  (15, True),
  (15, False)
]

inertias = {}
cluster_range = range(1, 11)
page = ('SELECT_names', 15)
for permutation in permutations:
  inertias[permutation] = []

  plottable_trajectories = [ trajectory for _, trajectory in trajectories[page].items() if trajectory is not None and len(trajectory) == permutation[0] and (trajectory[-1] == 0) == permutation[1] ]

  dataset = to_time_series_dataset(plottable_trajectories)

  for num_clusters in cluster_range:
    model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", max_iter=10, random_state=42)
    model.fit(dataset)
    inertias[permutation].append(model.inertia_)

plt.figure(figsize=(8, 10.5))
for idx, permutation in enumerate(permutations):
  plt.subplot(int(len(permutations) / 2), 2, idx + 1)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')
  plt.title(f'{'Successful' if permutation[1] else 'Failing'} trajectories of {permutation[0]} attempts')
  plt.xticks(cluster_range)
  plt.grid(True)
  for idx, _ in enumerate(permutations):
    plt.plot(cluster_range, inertias[permutation], marker='o', color='#1f77b4')

plt.tight_layout()
plt.show()

# %%

# Generate given number of clusters for trajectories of given lengths and result

ext_permutations = [
    (5, True, 5),
    (5, False, 4),
    (10, True, 5),
    (10, False, 4),
    (15, True, 3),
    (15, False, 4)
]

for permutation in ext_permutations:
  for page in [('SELECT_names', 15)]:
    plottable_trajectories = [ trajectory for _, trajectory in trajectories[page].items() if trajectory is not None and len(trajectory) == permutation[0] and (trajectory[-1] == 0) == permutation[1] ]

    dataset = to_time_series_dataset(plottable_trajectories)

    model = TimeSeriesKMeans(n_clusters=permutation[2], metric="dtw", max_iter=10, random_state=42)
    model.fit(dataset)

    clustered_data = [[] for _ in range(model.n_clusters)]
    for i, ts in enumerate(dataset):
        cluster_idx = model.labels_[i]
        clustered_data[cluster_idx].append(ts)

    plt.figure(figsize=(8, 5 * math.ceil(permutation[2] / 2)))
    plt.suptitle(f'{'Successful' if permutation[1] else 'Failing'} trajectories of {permutation[0]} attempts\n')
    cluster_inertia = []
    for cluster_idx, cluster_data in enumerate(clustered_data):
        plt.subplot(model.n_clusters, 2, cluster_idx + 1)
        plt.title(f'Cluster {cluster_idx + 1}, {len(cluster_data)} {'trajectories' if len(cluster_data) > 1 else 'trajectory'}')
        plt.xticks(np.arange(0, permutation[0], 1), [str(num) for num in range(1, permutation[0] + 1)])
        centroid = np.mean(cluster_data, axis=0)
        plt.plot(centroid.ravel(), 'r-', linewidth=2, label='Centroid')
        for idx, ts in enumerate(cluster_data):
          plt.plot(ts.ravel(), 'k-', alpha=.2)
        plt.legend()
        inertia = np.sum(np.square(model.transform(cluster_data)))
        cluster_inertia.append(inertia)
        plt.text(0.5, 0.95, f"Inertia: {inertia:.2f}", horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()
