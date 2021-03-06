
from sklearn import datasets
import pandas as pd
import graphviz as gv

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df

y = iris.target
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dtree = DecisionTreeClassifier()
dtree.fit(df, y)
dtree.tree_

dot_data = export_graphviz(dtree, out_file=None, feature_names=iris.feature_names,
                           filled=True, rounded=True, special_characters=True)
x = gv.Source(dot_data)
x.format ='png'
#x.format ='svg'
x.render('in pycharm, generate graph', view=True)