# %%

import mysql.connector
from globals import pages, ref_feature_subsets_with_aliases, ref_feature_subsets_with_common_alias, mssql_keywords
from utils import extract_query_features, provide_entities, cleanup
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

class FeatureBasedAttemptRecord:
  id: int
  username: str
  query_txt: str
  features: list[str] | None
  score: int | None

  def __init__(self, id: int, username: str, txt: str, features: list[str], score: int | None) -> None:
    self.id = id
    self.username = username
    self.query_txt = txt
    self.features = features
    self.score = score
  
  def __str__(self):
    return f'''
                ID: {self.id}
                USER NAME: {self.username}
                SCORE: {self.score}
                TXT LEN: {len(self.query_txt)}
                FEAT LEN: {len(self.features)}
            '''

# Calculate global alignment score. Returns 0 for perfect match.

def global_alignment_score(seq1, seq2, match=0, mismatch=1, gap=1):
    # Initialize matrix
    matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))

    # Fill in matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            match_score = matrix[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            gap1_score = matrix[i-1, j] + gap
            gap2_score = matrix[i, j-1] + gap
            matrix[i, j] = min(match_score, gap1_score, gap2_score)

    # Calculate alignment score
    alignment_score = matrix[-1, -1]

    return alignment_score

# Find a solution that is most similar to the supplied attempt.
# Unlike a similar method for optimized ASTs, this runs much faster.

def find_closest_ref_feature_set(page: tuple, features: list[str], with_aliases: bool) -> list[str]:
  ref_feature_set = set()
  if with_aliases:
    ref_feature_set = ref_feature_subsets_with_aliases[page]
  else:
    ref_feature_set = ref_feature_subsets_with_common_alias[page]

  least_alignment_score = 999999999
  ref_features = list[str]

  for subset in ref_feature_set:
    curr_alignment_score = global_alignment_score(features, subset)
    if curr_alignment_score < least_alignment_score:
      least_alignment_score = curr_alignment_score
      ref_features = subset
      if least_alignment_score == 1:
        break
  return ref_features

# %%

# Retrieve user names from database

users = {}

for page in pages:
  cursor.execute(f'SELECT DISTINCT(wgUserName) FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} ORDER BY wgUserName;')

  user_names = [ user_name[0] for user_name in cursor.fetchall() ]

  users[page] = user_names

# %%

# Retrieve and process user attempts into feature subsets of two types

start = time.time()
attempts_with_aliases = {}
attempts_with_common_alias = {}
for page in pages:
  print(f'Fetching for {page[0]}, {page[1]}...')
  ucount = 0
  attempts_with_aliases[page] = {}
  attempts_with_common_alias[page] = {}
  page_features = mssql_keywords + provide_entities(page[0])
  cursor.execute(f'SELECT id, wgUserName, txt, score FROM gisqlog WHERE page = "{page[0]}" AND qnum = {page[1]} AND wgUserName IN (\'{"\', \'".join(users[page])}\') ORDER BY wgUserName, id')
  response = cursor.fetchall()
  for username in users[page]:
    ucount += 1
    if (ucount % 100 == 0):
      print(f'Processed {ucount} users for this page.')
      print(f'Time elapsed: {round((time.time() - start) / 60 / 60, 2)} hours.')
    records = [ record for record in response if record[1] == username ]
    attempts_with_aliases[page][username] = [ FeatureBasedAttemptRecord(record[0], record[1], record[2], list(extract_query_features(cleanup(record[2]), page_features)), record[3]) for record in records if record[2].strip() != '' ]
    attempts_with_common_alias[page][username] = [ FeatureBasedAttemptRecord(record[0], record[1], record[2], list(extract_query_features(cleanup(record[2]), page_features, True, True)), record[3]) for record in records if record[2].strip() != '' ]

# %%

# Store reference solution for each user
# This is either the user's own solution if they reached it,
# or a solution most similar to their last attempt.

ref_solutions_for_subset_with_aliases = {}
ref_solutions_for_subset_with_common_alias = {}

start = time.time()

for page in pages:
  ucount = 0
  idx_errs_with_aliases = 0
  idx_errs_with_common_alias = 0
  print(f'Processing ref solutions for {page[0]}, question {page[1]}...')
  ref_solutions_for_subset_with_aliases[page] = {}
  ref_solutions_for_subset_with_common_alias[page] = {}
  for username in users[page]:
    ucount += 1
    if (ucount % 100 == 0):
      print(f'Processed {ucount} users for this page.')
      print(f'Time elapsed: {round((time.time() - start) / 60 / 60, 2)} hours.')
    
    try:
      last_attempt_with_aliases: FeatureBasedAttemptRecord = attempts_with_aliases[page][username][-1]
      if last_attempt_with_aliases.score == 100:
        ref_solutions_for_subset_with_aliases[page][username] = last_attempt_with_aliases.features
      else:
        ref_solutions_for_subset_with_aliases[page][username] = find_closest_ref_feature_set(page, last_attempt_with_aliases.features, True)
    except IndexError:
      idx_errs_with_aliases += 1
      continue

    try:
      last_attempt_with_common_alias: FeatureBasedAttemptRecord = attempts_with_common_alias[page][username][-1]
      if last_attempt_with_common_alias.score == 100:
        ref_solutions_for_subset_with_common_alias[page][username] = last_attempt_with_common_alias.features
      else:
        ref_solutions_for_subset_with_common_alias[page][username] = find_closest_ref_feature_set(page, last_attempt_with_common_alias.features, False)
    except IndexError:
      idx_errs_with_common_alias += 1
      continue

# %%

# Generate trajectories for feature sets with aliases preserved and with a common alias.

start = time.time()

trajectories_with_aliases = {}
trajectories_with_common_alias = {}

key_errs_with_aliases = 0
key_errs_with_common_alias = 0

for page in pages:
  print(f'Tracing trajectories for {page[0]}, question {page[1]}...')
  ucount = 0
  trajectories_with_aliases[page] = {}
  trajectories_with_common_alias[page] = {}
  for username in users[page]:
    ucount += 1
    if (ucount % 100 == 0):
      print(f'Processed {ucount} users for this page.')
      print(f'Time elapsed: {round((time.time() - start) / 60 / 60, 2)} hours.')

    try:
      reference_with_aliases = ref_solutions_for_subset_with_aliases[page][username]
    except KeyError:
      key_errs_with_aliases += 1
      print(f'TOTAL KEY ERRORS WITH ALIASES: {key_errs_with_aliases}')
    
    try:
      reference_with_common_alias = ref_solutions_for_subset_with_common_alias[page][username]
    except KeyError:
      key_errs_with_common_alias += 1
      print(f'TOTAL KEY ERRORS WITH COMMON ALIAS: {key_errs_with_common_alias}')
    
    if reference_with_aliases is None:
      trajectories_with_aliases[page][username] = None
    else:
      records_with_aliases: list[FeatureBasedAttemptRecord] = [ attempt for attempt in attempts_with_aliases[page][username] ]
      distances_with_aliases = [ global_alignment_score(record.features, reference_with_aliases) for record in records_with_aliases ]
      trajectories_with_aliases[page][username] = distances_with_aliases

    if reference_with_common_alias is None:
      trajectories_with_common_alias[page][username] = None
    else:
      records_with_common_alias: list[FeatureBasedAttemptRecord] = [ attempt for attempt in attempts_with_common_alias[page][username] ]
      distances_with_common_alias = [ global_alignment_score(record.features, reference_with_common_alias) for record in records_with_common_alias ]
      trajectories_with_common_alias[page][username] = distances_with_common_alias

# %%

# Store obtained trajectories in database.

sql = 'INSERT INTO trajectories_with_common_alias (pg, qnum, username, trajectory) VALUES (%s, %s, %s, %s)'
for page in pages:
  insert: list[tuple] = []
  print(f'Processing page {page[0]}...')
  for username in users[page]:
    trajectory = trajectories_with_common_alias[page][username]
    if trajectory is not None:
      trajectory = ', '.join([str(int(dist)) for dist in trajectory])
    insert.append((page[0], page[1], username, trajectory))

  btm = 0
  top = 1000
  while btm < len(insert):
    cursor.executemany(sql, insert[btm:top])
    connector.commit()
    btm += 1000
    top += 1000
    print(f'Inserted {cursor.rowcount} rows.')

# %%

# Produce bar charts of trajectory distributions by length by question

lengths = {}
for page in pages:
  lengths[page] = {}
  for idx in range(1, 31):
    lengths[page][f'Solved in {idx}'] = 0
    lengths[page][f'Unsolved in {idx}'] = 0
  for username in users[page]:
    trajectory = trajectories_with_common_alias[page][username]
    if trajectory is not None and len(trajectory) > 0 and len(trajectory) < 31:
      if (trajectory[-1] == 0.0):
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
  (15, False),
  (20, True),
  (20, False)
]

inertias = {}
cluster_range = range(1, 21)
page = ('The_JOIN_operation', 13)
for permutation in permutations:
  inertias[permutation] = []

  plottable_trajectories = [ trajectory for _, trajectory in trajectories_with_common_alias[page].items() if trajectory is not None and len(trajectory) == permutation[0] and (trajectory[-1] == 0) == permutation[1] ]

  dataset = to_time_series_dataset(plottable_trajectories)

  for num_clusters in cluster_range:
    model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", max_iter=10, random_state=42)
    model.fit(dataset)
    inertias[permutation].append(model.inertia_)

plt.figure(figsize=(8, 10.5))
plt.suptitle(f'Elbow plots for question {page[1]} in {' '.join(page[0].split('_'))}')
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

ext_permutations_for_subsets_with_aliases = [
  (5, True, 4),
  (5, False, 3),
  (10, True, 4),
  (10, False, 4),
  (15, True, 4),
  (15, False, 4),
  (20, True, 4),
  (20, False, 2)
]

ext_permutations_for_subsets_with_common_alias = [
  (5, True, 4),
  (5, False, 5),
  (10, True, 3),
  (10, False, 5),
  (15, True, 4),
  (15, False, 4),
  (20, True, 5),
  (20, False, 3)
]

page = ('The_JOIN_operation', 13)

for permutation in ext_permutations_for_subsets_with_aliases:
  plottable_trajectories = [ trajectory for _, trajectory in trajectories_with_common_alias[page].items() if trajectory is not None and len(trajectory) == permutation[0] and (trajectory[-1] == 0) == permutation[1] ]

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
  plt.suptitle(f'{'Successful' if permutation[1] else 'Failing'} trajectories of {permutation[0]} attempts with aliases preserved\n')
  cluster_inertia = []
  for cluster_idx, cluster_data in enumerate(clustered_data):
      plt.subplot(model.n_clusters, 2, cluster_idx + 1)
      plt.title(f'Cluster {cluster_idx + 1}, {len(cluster_data)} {'trajectories' if len(cluster_data) > 1 else 'trajectory'}')
      plt.xticks(np.arange(0, permutation[0], 1), [str(num) for num in range(1, permutation[0] + 1)])
      centroid = np.mean(cluster_data, axis=0)
      plt.plot(centroid.ravel(), 'r-', linewidth=2)
      for idx, ts in enumerate(cluster_data):
        plt.plot(ts.ravel(), 'k-', alpha=.2)
      inertia = np.sum(np.square(model.transform(cluster_data)))
      cluster_inertia.append(inertia)
      plt.text(0.5, 0.95, f"Inertia: {inertia:.2f}", horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
  plt.tight_layout()
  plt.show()
# %%
