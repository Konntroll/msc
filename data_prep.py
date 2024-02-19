# %%

import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from matplotlib.ticker import FuncFormatter

# %%

connector = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Zoo"
)

cursor = connector.cursor()

# %%

# PERFORM INITIAL DATA CLEANUP

cursor.execute('DELETE FROM gisqlog WHERE lang <> "sql";')
connector.commit()

cursor.execute("""
  DELETE FROM gisqlog 
  WHERE machine NOT IN ('mssql', 'mysql', 'postgres') 
    OR machine IS NULL 
    OR wgUserName IS NULL;
""")
connector.commit()

cursor.execute('UPDATE gisqlog SET wgUserName = NULL WHERE TRIM(wgUserName) = '';')
connector.commit()

cursor.execute("""
  DELETE FROM gisqlog 
  WHERE wgUserName IN ('1  /*!50000AND*/ (/*!50000SeLect*/ 6556 /*!50000Fr', '1 AND 6556=(SELECT 6556 FROM PG_SLEEP(16)) AND \'aq', '1 OR 6*1*5=9*1 -- ', '1" /*!50000AND*/ (/*!50000SeLect*/ 6556 /*!50000Fr', '1" AND (SELECT 6556 FROM (SELECT(SLEEP(16)))aqkh)-', '1"XOR(if(now()=sysdate(),sleep(16),0))XOR"Z', '1\' /*!50000AND*/ (/*!50000SeLect*/ 6556 /*!50000Fr', '1\' OR 6556=(SELECT 6556 FROM PG_SLEEP(16))--', '1;WAITFOR DELAY \'0:0:16\'--', '\'"<img src=x onerror=prompt()>') 
    OR wgUserName LIKE '%#EXEC cmd=%';
""")
connector.commit()

cursor.execute('UPDATE gisqlog SET wgUserName = "john12342" WHERE wgUserName = '"john12342"'')

cursor.execute("""
  ALTER TABLE gisqlog
  DROP COLUMN `engine`,
  DROP COLUMN loip,
  DROP COLUMN question,
  DROP COLUMN lang,
  DROP COLUMN extime;
""")

# %%

# MERGE USER NAMES THAT APPEAR TO BELONG TO THE SAME USER

shared_cookies = ['1255582198', '1269419056', '1483971373', '1576517187', '1608926156']

for idx, cookie in enumerate(shared_cookies):
  alias = 'TempNU' + str(idx + 1)
  cursor.execute(f"""
                  UPDATE gisqlog 
                  SET wgUserName = "{alias}" 
                  WHERE cookie = "{cookie}" 
                    AND wgUserName IS NOT NULL;
                 """)
  
# %%

# GET COOKIE IDs WHERE THE COOKIE WITH ONE-TO-ONE RELATIONSHIP TO A USER

cursor.execute("""
  SELECT DISTINCT(cookie)
  FROM gisqlog
  WHERE cookie IN (
    SELECT cookie
    FROM gisqlog
    WHERE wgUserName IS NOT NULL
      AND wgUserName IN (
        SELECT wgUserName
        FROM gisqlog
        WHERE wgUserName IS NOT NULL
        GROUP BY wgUserName
        HAVING COUNT(DISTINCT(cookie)) = 1
      )
      AND cookie IN (
        SELECT cookie
        FROM gisqlog
        WHERE wgUserName IS NOT NULL
        GROUP BY cookie
        HAVING COUNT(DISTINCT(wgUserName)) = 1
      )
    GROUP BY cookie
  );
""")

cookies = [cookie[0] for cookie in cursor.fetchall()]

one_to_ones = len(cookies)

# %%

# GET THE NUMBER OF NAMED RECORDS PER COOKIE WITH ONE-TO-ONE RELATIONSHIP TO A USER

cursor.execute("""
  SELECT COUNT(*) records
  FROM gisqlog
  WHERE wgUserName IS NOT NULL
    AND wgUserName IN (
      SELECT wgUserName
      FROM gisqlog
      WHERE wgUserName IS NOT NULL
      GROUP BY wgUserName
      HAVING COUNT(DISTINCT(cookie)) = 1
    )
    AND cookie IN (
      SELECT cookie
      FROM gisqlog
      WHERE wgUserName IS NOT NULL
      GROUP BY cookie
      HAVING COUNT(DISTINCT(wgUserName)) = 1
    )
  GROUP BY cookie
  ORDER BY records DESC;
""")

named = np.array([cookie[0] for cookie in cursor.fetchall()])

# %%

# GET THE NUMBER OF NAMED AND UNNAMED RECORDS PER COOKIE WITH ONE-TO-ONE RELATIONSHIP TO A USER

cursor.execute("""
  SELECT COUNT(*) records
  FROM gisqlog
  WHERE cookie IN (
    SELECT cookie
    FROM gisqlog
    WHERE wgUserName IS NOT NULL
      AND wgUserName IN (
        SELECT wgUserName
        FROM gisqlog
        WHERE wgUserName IS NOT NULL
        GROUP BY wgUserName
        HAVING COUNT(DISTINCT(cookie)) = 1
      )
      AND cookie IN (
        SELECT cookie
        FROM gisqlog
        WHERE wgUserName IS NOT NULL
        GROUP BY cookie
        HAVING COUNT(DISTINCT(wgUserName)) = 1
      )
    GROUP BY cookie
  )
  GROUP BY cookie
  ORDER BY records DESC;
""")

extended = np.array([cookie[0] for cookie in cursor.fetchall()])

# %%

# CREATE BOXPLOTS SHOWING DISTRIBUTION OF RECORDS BY COOKIE/USER

_, axs = plt.subplots(1, 3, figsize=(10, 12))

axs[0].boxplot(named)
axs[0].set_title('With outliers')

axs[1].boxplot(named, 0, '')
axs[1].set_title('Without outliers')
axs[1].set_xticklabels([2])

axs[2].boxplot(extended, 0, '')
axs[2].set_title('Extended, no outliers')
axs[2].set_xticklabels([3])

plt.show()

# see Fig. 3

# %%

# GET THE EXTENDED SET'S STATS: Q1, Q1, IQR, SD, mean, median

q3, q1 = np.percentile(extended, [75 ,25])
print('Q1:', q1)
print('Q3:', q3)
print('IQR:', q3 - q1)
print('SD:', np.std(extended))
print('Mean', np.mean(extended))
print('Median', np.median(extended))

# Q1: 17.0
# Q3: 198.5
# IQR: 181.5
# SD: 218.3286537005523
# Mean 151.2508765095442
# Median 77.0

# %%

# FILTER THE EXTENDED SET AND GET UPDATED STATS: fQ1, fQ1, fIQR, fSD, fMean, fMedian

filtered = [x for x in extended if x > 50 and x < 526]

fq3, fq1 = np.percentile(filtered, [75 ,25])
print('fQ1:', fq1)
print('fQ3:', fq3)
print('fIQR:', fq3 - fq1)
print('fSD:', np.std(filtered))
print('fMean', np.mean(filtered))
print('fMedian', np.median(filtered))

# fQ1: 96.0
# fQ3: 254.5
# fIQR: 158.5
# fSD: 118.92286910339541
# fMean 190.890350877193
# fMedian 157.0

# %%

# PLOT DISTRIBUTION OF COMBINED RECORD NUMBERS

sns.displot(filtered)

# see Fig. 4

# %%

# RENAME RECORDS ASSOCIATED WITH COOKIES WITH ONE-TO-ONE RELATIONSHIP TO A USER

for idx, cookie in enumerate(cookies):
  print(f'Processing record ({idx + 1}/{len(cookies)}).')
  alias = 'NU' + str(idx + 1)
  cursor.execute(f'UPDATE gisqlog SET wgUserName = "{alias}" WHERE cookie = "{cookie}"')
  connector.commit()

# %%

# RENAME OTHER NAMED USERS

cursor.execute('SELECT DISTINCT(wgUserName) FROM gisqlog WHERE wgUserName NOT LIKE "NU%" AND wgUserName IS NOT NULL;')

users = [user[0] for user in cursor.fetchall()]

for idx, user in enumerate(users):
  print(f'Processing record ({idx + 1}/{len(users)}).')
  alias = 'NU' + str(one_to_ones + idx + 1)
  cursor.execute(f'UPDATE gisqlog SET wgUserName = "{alias}" WHERE wgUserName = "{user}"')
  connector.commit()

# %%

# TAG ASSUMED USERS, SKIP COOKIES THAT ARE SHARED BETWEEN NAMED USERS
  
cursor.execute("""
  SELECT cookie
  FROM gisqlog
  WHERE wgUserName IS NULL
    AND cookie NOT IN ('1092801048', '1229158381', '1672527121', '1708607105', '2138465819', '261440179', '289688697', '299726987', '470627442', '511949793', '649898598', '901970927', '929647024', 'b0e253a184cd099f31')
  GROUP BY cookie
  HAVING COUNT(*) >= 72
     AND COUNT(*) <= 310;
""")

unnamed_cookies = [cookie[0] for cookie in cursor.fetchall()]

print('NUMBER OF ASSUMED USERS:', len(unnamed_cookies)) # 32475

# %%

for idx, cookie in enumerate(unnamed_cookies):
  print(f'Processing record ({idx + 1}/{len(unnamed_cookies)}).')
  alias = 'AU' + str(idx + 1)
  cursor.execute(f'UPDATE gisqlog SET wgUserName = "{alias}" WHERE cookie = "{cookie}"')
  connector.commit()
# %%

# FILTER OUT RECORDS WITH QUERIES THAT ARE NOT MADE AGAINST MS SQL SERVER AND THE REMAINING UNNAMED RECORDS

cursor.execute('DELETE FROM gisqlog WHERE machine <> "mssql" OR wgUserName IS NULL;')
connector.commit()

# %%

# FOR EACH USER, REMOVE ALL RECORDS FOR A GIVEN QUESTION IN A GIVEN CATEGORY PAST THE FIRST TO CONTAIN A PASSING QUERY

class Record:
  id: int
  page: str
  qnum: int
  score: int

  def __init__(self, id, page, qnum, score):
    self.id = id
    self.page = page
    self.qnum = qnum
    self.score = score
  
  def is_not_in(self, records: list['Record']):
    return len(list(filter(lambda r: r.page == self.page and r.qnum == self.qnum, records))) == 0

start = time.time()

cursor.execute('SELECT DISTINCT(wgUserName) FROM gisqlog;')

users = [user[0] for user in cursor.fetchall()]

for idx, user in enumerate(users):
  print(f'Processing records for user {idx + 1} of {len(users)}.')
  cursor.execute(f'SELECT id, page, qnum, score FROM gisqlog WHERE wgUserName = "{user}" ORDER BY id, page, qnum;')
  records = [Record(*record) for record in cursor.fetchall()]

  if len(records) <= 1:
    continue

  first_passing_instances: list['Record'] = []

  for record in records:
    if record.score == 100 and record.is_not_in(first_passing_instances):
      first_passing_instances.append(record)
  
  for record in first_passing_instances:
    cursor.execute(f"""
        DELETE FROM gisqlog
        WHERE id > {record.id}
          AND page = "{record.page}"
          AND qnum = "{record.qnum}"
          AND wgUserName = "{user}";
      """)
    connector.commit()

end = time.time()

print(f'Processing concluded in {round((end - start) / 60 / 60, 2)} hours.')

# %%

# RATE CATEGORIES BY THE NUMBER OF ATTEMPTS

cursor.execute("""
  SELECT `page`, COUNT(`page`) records
  FROM gisqlog
  GROUP BY `page`
  ORDER BY records DESC;
""")

main_cats = cursor.fetchall()

def format_millions(value, _):
    return f'{round(value / 1000000, 1)}M'

formatter = FuncFormatter(format_millions)

main_categories = [(item[0].replace('_', ' ')[0:20] + ('...' if len(item[0]) > 20 else ''), item[1]) for item in main_cats[0:15]]
main_categories_df = pd.DataFrame(main_categories, columns=['Category', 'Records'])
ax = main_categories_df.plot(x='Category', y='Records', kind='bar')
plt.xticks(range(0, 15))
ax.yaxis.set_major_formatter(formatter)
plt.title('Main categories')
plt.show()

# %%

for cat in main_cats[0:6]:
  print(cat[0])

# %%

# DELETE RECORDS WITH QUESTION NUMBERS 0 AND 21 AS THESE DON'T CORRESPOND TO ACTUAL PROBLEMS IN SQLZoo

cursor.execute('DELETE FROM gisqlog WHERE qnum = 0 OR qnum = 21;')
connector.commit()

# %%

# CREATE BOXPLOTS FOR THE NUMBER OF ATTEMPTS MADE BY INDIVIDUAL USERS AT EACH QUESTION IN THE TOP 6 CATEGORIES
# CREATE BAR CHARTS FOR THE NUMBER OF UNIQUE USERS ATTEMPTING EACH QUESTION IN A GIVEN CATEGORY

for cat in [name[0] for name in main_cats[0:6]]:
  cursor.execute(f'SELECT qnum, COUNT(*) attempts FROM gisqlog WHERE page = "{cat}" GROUP BY wgUserName, qnum;')
  cat_breakdown = cursor.fetchall()

  cat_attepts_by_q_df = pd.DataFrame(data=cat_breakdown, columns=['Question', 'Attempts'])

  plt.figure(figsize=(9, 5))
  plt.boxplot(cat_attepts_by_q_df.groupby('Question')['Attempts'].apply(list).values, 0, '', labels=np.sort(cat_attepts_by_q_df['Question'].unique()))
  plt.xlabel('Question')
  plt.ylabel('Number of Attempts')
  plt.show()

  cursor.execute(f'SELECT qnum, COUNT(DISTINCT(wgUserName)) FROM gisqlog WHERE page = "{cat}" AND machine = "mssql" GROUP BY qnum ORDER BY qnum;')
  users_per_question = cursor.fetchall()

  users_per_question_df = pd.DataFrame(data=users_per_question, columns=['Question', 'Users'])

  plt.figure(figsize=(9, 5))
  plt.bar(np.sort(users_per_question_df['Question']), users_per_question_df['Users'], label=users_per_question_df['Question'])
  plt.xticks(np.sort(users_per_question_df['Question']))
  plt.xlabel('Question')
  plt.ylabel('Users')
  plt.show()

# %%

class Category:
  name: str
  qnums: list

  def __init__(self, name, qnums):
    self.name = name
    self.qnums = qnums

cats_of_interest: list['Category'] = []

cats_of_interest.append(Category('SELECT_from_WORLD_Tutorial', [8, 10, 11, 13]))
cats_of_interest.append(Category('SELECT_from_Nobel_Tutorial', [14]))
cats_of_interest.append(Category('The_JOIN_operation', [8, 11, 13]))
cats_of_interest.append(Category('SELECT_within_SELECT_Tutorial', [5]))
cats_of_interest.append(Category('SELECT_names', [12, 15]))
cats_of_interest.append(Category('More_JOIN_operations', [6, 12, 13, 14, 15]))

for cat in cats_of_interest:
  for qnum in cat.qnums:
    cursor.execute(f'SELECT COUNT(*) FROM gisqlog WHERE page = "{cat.name}" AND qnum = {qnum} GROUP BY wgUserName;')
    counts = cursor.fetchall()

    print(f'Category {cat.name}, question {qnum}:')
    q3, q1 = np.percentile(counts, [75 ,25])
    print('Q1:', q1)
    print('Q3:', q3)
    print('IQR:', q3 - q1)
    print('SD:', round(np.std(counts), 2))
    print('Mean', round(np.mean(counts), 2))
    print('Median', np.median(counts))

    cursor.execute(f'SELECT COUNT(*) FROM gisqlog WHERE page = "{cat.name}" AND qnum = {qnum};')
    total_attempts = cursor.fetchall()[0][0]

    cursor.execute(f'SELECT COUNT(DISTINCT(wgUserName)) FROM gisqlog WHERE page = "{cat.name}" AND qnum = {qnum};')
    total_individual_users = cursor.fetchall()[0][0]

    cursor.execute(f'SELECT COUNT(DISTINCT(wgUserName)) FROM gisqlog WHERE page = "{cat.name}" AND qnum = {qnum} AND score = 100;')
    total_individual_users_who_succeeded = cursor.fetchall()[0][0]

    print(f'Total attempts: {total_attempts}')
    print(f'Total users: {total_individual_users}')
    print(f'Users succeeded: {total_individual_users_who_succeeded}')
    print(f'Success ratio: {round(total_individual_users_who_succeeded / total_individual_users, 2)}')


# %%
