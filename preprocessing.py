import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('nursery.csv')

# parents = {'usual': 1, 'pretentious': 2, 'great_pret': 3}
# data.parents = [parents[item] for item in data.parents]
#
# has_nurs = {'proper': 1, 'less_proper': 2, 'improper': 3, 'critical': 4, 'very_crit': 5}
# data.has_nurs = [has_nurs[item] for item in data.has_nurs]
#
# form = {'complete': 1, 'completed': 2, 'incomplete': 3, 'foster ': 4}
# data.form = [form[item] for item in data.form]
#
# children = {1: 1, 2: 2, 3: 3, 'more': 4}
# data.children = [children[item] for item in data.children]
#
# housing = {'convenient': 1, 'less_conv': 2, 'critical ': 3}
# data.housing = [housing[item] for item in data.housing]
#
# finance = {'convenient': 1, 'inconv ': 2}
# data.finance = [finance[item] for item in data.finance]
#
# social = {'non-prob': 1, 'slightly_prob': 2, 'problematic ': 3}
# data.social = [social[item] for item in data.social]
#
# health = {'recommended': 1, 'priority': 2, 'not_recom': 3}
# data.health = [health[item] for item in data.health]

le = LabelEncoder()

data['parents'] = le.fit_transform(data['parents'])
data['has_nurs'] = le.fit_transform(data['has_nurs'])
data['form'] = le.fit_transform(data['form'])
data['children'] = le.fit_transform(data['children'])
data['housing'] = le.fit_transform(data['housing'])
data['finance'] = le.fit_transform(data['finance'])
data['social'] = le.fit_transform(data['social'])
data['health'] = le.fit_transform(data['health'])
data['class'] = le.fit_transform(data['class'])

data.to_csv('nursery_processed.csv', index = False)
