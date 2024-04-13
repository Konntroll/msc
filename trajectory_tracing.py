# %%

import mysql.connector
import sqlparse
import copy
from sqlglot.optimizer import optimize
from sqlglot import Expression, ParseError, parse_one, diff
from globals import pages, optimized_asts_for_passing_queries, Errors
from utils import provide_schema, catch

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

connector.close()

# %%

cursor.execute(f'SELECT DISTINCT(wgUserName) FROM gisqlog WHERE page = "SELECT_within_SELECT_Tutorial" AND qnum = 5 AND score = 100;')

user_names = cursor.fetchall()

print(user_names)

# %%

cursor.execute(f'SELECT txt FROM gisqlog WHERE page = "SELECT_within_SELECT_Tutorial" AND qnum = 5 AND wgUserName = "AU4286" AND errno IS NULL ORDER BY id')

queries = [ query[0] for query in cursor.fetchall() ]

clean_queries = [ query.replace('ã€€', '').replace('Â', '').replace('< =', '<=').replace('> =', '>=').replace('?', '').replace('', '').strip() for query in queries ]

print(queries[0])

# %%

asts = [ parse_one(query, dialect='tsql') for query in clean_queries ]

schema = provide_schema("SELECT_within_SELECT_Tutorial")

opts = [ optimize(ast, schema=schema, dialect='tsql') for ast in asts if catch(optimize, ast, schema=schema, dialect='tsql') != None ]

diff_res = diff(opts[0], opts[1])

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

  # print(f'Keeps: {keeps}, Removes: {removes}, Inserts: {inserts}, Moves: {moves}, Updates: {updates}, Total: {total}')

  return total

# %%

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
comparisons = 0
for (idx, opt) in enumerate(opts):
  if (idx + 1 < len(opts)):
    comparisons += 1
    get_edit_distance(diff(opt, opts[-1]))

print(f'Comparisons: {comparisons}')
# %%
import numpy as np

def global_alignment_score(seq1, seq2, match=1, mismatch=-1, gap=-1):
    # Initialize matrix
    matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))

    # Fill in matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            match_score = matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            gap1_score = matrix[i-1, j] + gap
            gap2_score = matrix[i, j-1] + gap
            matrix[i, j] = max(match_score, gap1_score, gap2_score)

    # Calculate alignment score
    alignment_score = matrix[-1, -1]

    return alignment_score

# Example usage
seq1 = ['SELECT', 'name', 'FROM', 'the_table', 'WHERE', 'id', '10']
seq2 = ['SELECT', 'name', 'AS', 'Usernames', 'FROM', 'the_table', 'WHERE', 'id', '0']
score = global_alignment_score(seq1, seq2)
print("Global alignment score:", score)

# %%

for page in pages:
  print(f'{page[0]} length: {len(optimized_asts_for_passing_queries[page])}')

# %%

users = {}

for page in pages:
  cursor.execute(f'SELECT DISTINCT(wgUserName) FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} ORDER BY wgUserName;')

  user_names = [ user_name[0] for user_name in cursor.fetchall() ]

  users[page] = user_names

# %%

for page in pages:
  print(f'{page[0]} users: {len(users[page])}')

# %%

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
  for username in users[page][0:5]:
    try:
      print(f'Exp for {username} on {page[0]}: {type(ref_solutions_by_page_by_user[page][username])}')
    except KeyError:
      continue

# %%

for page in pages:
  page_errors: Errors = errors_by_page[page]
  print(f'Errors for {page[0]}, {page[1]}:')
  print(f'Attempts for page: {len(attempts[page])}')
  print(f'Parse errors: {page_errors.parse_errors}')
  print(f'Key errors: {page_errors.key_errors}')
  print(f'Other errors: {page_errors.other_errors}')

# %%

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

features: list[list] = []
for username in users[('SELECT_from_WORLD_Tutorial', 8)][0:100]:
  solutions = [ record for record in attempts[('SELECT_from_WORLD_Tutorial', 8)] if record.score == 100 and record.username == username ]
  has_solution = False
  if (len(solutions) > 0):
    has_solution = solutions[-1].score == 100
  trajectory_for_user = trajectories[('SELECT_from_WORLD_Tutorial', 8)][username]
  if trajectory_for_user is not None:
    features.append([len(trajectory_for_user), np.std(trajectory_for_user), has_solution])

features_df = pd.DataFrame(features, columns=['attempts', 'std_deviation', 'found_solution'])

dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed

# Fit and predict clusters
clusters = dbscan.fit_predict(features_df[['attempts', 'std_deviation', 'found_solution']])

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='attempts', y='found_solution', hue=clusters, data=features_df, palette='Set1', legend='full')
plt.title('DBSCAN Clustering')
plt.xlabel('Number of Attempts')
plt.ylabel('Standard Deviation')
plt.show()

# %%

features: list[list] = []
for username in users[('SELECT_from_WORLD_Tutorial', 8)]:
  trajectory_for_user = trajectories[('SELECT_from_WORLD_Tutorial', 8)][username]
  if trajectory_for_user is not None:
    has_solution = trajectory_for_user[-1] == 0
    std = np.std(trajectory_for_user)
    tlen = len(trajectory_for_user)
    if std > 5 or tlen > 10:
      continue
    else:
      features.append([len(trajectory_for_user), np.std(trajectory_for_user), has_solution])

features_df = pd.DataFrame(features, columns=['attempts', 'std_deviation', 'found_solution'])

scaler = StandardScaler()
features_df_scaled = scaler.fit_transform(features_df)

kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(features_df_scaled)

features_df_scaled_df = pd.DataFrame(features_df_scaled, columns=['attempts', 'std_deviation', 'found_solution'])

# Visualize the clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='attempts', y='std_deviation', hue=clusters, data=features_df_scaled_df, palette='Set1', legend='full',  style='found_solution', markers=['o', 's'],)
plt.title('KMeans Clustering')
plt.xlabel('Number of Attempts')
plt.ylabel('Standard Deviation')
plt.show()

# %%

lists_of_optimized_asts_for_passing_queries = {}
for page in pages:
  lists_of_optimized_asts_for_passing_queries[page] = [ ast for ast in optimized_asts_for_passing_queries ]

rows: list[tuple] = [()]
for page in pages:
  for username in users[page]:
    user_attempts = [ attempt for attempt in attempts[page] if attempt.username == username and attempt.score == 100 ]
    if user_attempts is not None:
      continue
    user_attempts = [ attempt for attempt in attempts[page] if attempt.username == username and attempt.errno is None ]
    if user_attempts is None:
      continue
    ref = ref_solutions_by_page_by_user[page][username]
    ref_idx = lists_of_optimized_asts_for_passing_queries[page].index(ref)
    rows.append(ref_idx, page[0], page[1])

# %%

connector = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Zoo"
)

cursor = connector.cursor()

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

connector = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Zoo"
)

cursor = connector.cursor()

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
print(str(ref_solutions_by_page_by_user[('SELECT_from_WORLD_Tutorial', 8)]['AU1000']))
# %%

# Sample observations
observations = [
    [8, [19, 3, 3, 4, 3, 5, 3, 0]],
    [2, [12, 17]],
    [5, [7, 4, 4, 5, 0]],
    [1, [0]],
    [9, [11, 11, 23, 14, 17, 3, 4, 15, 9]],
    [6, [6, 5, 5, 4, 2, 0]],
    [3, [12, 14, 10]],
    [6, [7, 6, 4, 4, 5, 3]]
]

dataset_df = pd.DataFrame(observations, columns=['attempts', 'distance'])
dataset = to_time_series_dataset(dataset_df)
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
clusters = model.fit_predict(dataset_df)

# model.cluster_centers_.shape

# Visualize the dendrogram
plt.figure(figsize=(10, 5))
sns.scatterplot(x='attempts', y='std_deviation', hue=clusters, data=dataset_df, palette='Set1', legend='full')
plt.title('Hierarchical Clustering Dendrogram (with DTW)')
plt.xlabel('Observation Index')
plt.ylabel('Distance')
plt.show()

# %%

lengths = {}
for page in pages:
  lengths[page] = {}
  for username in users[page]:
    trajectory = trajectories[page][username]
    if trajectory is not None and len(trajectory) < 46:
      try:
        lengths[page][len(trajectory)] += 1
      except KeyError:
        lengths[page][len(trajectory)] = 1

for page in pages:
  keys = list(lengths[page].keys())
  values = list(lengths[page].values())

  plt.figure(figsize=(12, 8))
  plt.bar(keys, values)
  plt.xticks(range(min(keys), max(keys) + 1))
  plt.xlabel('Trajectory length')
  plt.ylabel('Trajectories of given length')
  plt.title(f'{page[0].split('_')}, question {page[1]}')
  plt.show()

# %%

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

for page in [('SELECT_from_WORLD_Tutorial', 8)]:
  rng = 5
  plottable_trajectories = [ trajectory for _, trajectory in trajectories[page].items() if trajectory is not None and len(trajectory) == rng and trajectory[-1] == 0 ]

  # Convert observations to time series dataset
  dataset = to_time_series_dataset(plottable_trajectories)

  # Cluster the time series data using TimeSeriesKMeans
  model = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=10, random_state=42)
  model.fit(dataset)

  # Separate time series data points by cluster
  clustered_data = [[] for _ in range(model.n_clusters)]
  for i, ts in enumerate(dataset):
      cluster_idx = model.labels_[i]
      clustered_data[cluster_idx].append(ts)

  # Plot each cluster separately
  plt.title(f'{' '.join(page[0].split('_'))}, question {page[1]}')
  plt.figure(figsize=(12, 15))
  cluster_inertia = []
  for cluster_idx, cluster_data in enumerate(clustered_data):
      plt.subplot(model.n_clusters, 2, cluster_idx + 1)
      plt.title(f'Cluster {cluster_idx + 1}, elements {len(cluster_data)}')
      plt.xticks(np.arange(0, rng, 1), [str(num) for num in range(1, rng + 1)])
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

# %%

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

    # Convert observations to time series dataset
    dataset = to_time_series_dataset(plottable_trajectories)

    # Cluster the time series data using TimeSeriesKMeans
    model = TimeSeriesKMeans(n_clusters=permutation[2], metric="dtw", max_iter=10, random_state=42)
    model.fit(dataset)

    # Separate time series data points by cluster
    clustered_data = [[] for _ in range(model.n_clusters)]
    for i, ts in enumerate(dataset):
        cluster_idx = model.labels_[i]
        clustered_data[cluster_idx].append(ts)

    # Plot each cluster separately
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

# %%
  
for page in [('SELECT_names', 15)]:
  inertia_values = []
  cluster_range = range(1, 21)

  plottable_trajectories = [ trajectory for _, trajectory in trajectories[page].items() if trajectory is not None and len(trajectory) == 10 and trajectory[-1] == 0 ]

  dataset = to_time_series_dataset(plottable_trajectories)

  for num_clusters in cluster_range:
    model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", max_iter=10, random_state=1984)
    model.fit(dataset)
    inertia_values.append(model.inertia_)

  plt.plot(cluster_range, inertia_values, marker='o')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')
  plt.title('Successful trajectories of 10 attempts')
  plt.xticks(cluster_range)
  plt.grid(True)
  plt.show()