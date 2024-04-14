# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from IPython.display import Image

sns.set(font_scale=1.3)
np.random.seed(42)

# Załadowanie danych
raw_data = load_iris()
all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']
feature_names = [name.replace(' ', '_')[:-5] for name in all_data['feature_names']]
target_names = all_data['target_names']

print(f'Liczba próbek: {len(data)}')
print(f'Kształt danych: {data.shape}')
print(f'Nazwy zmiennych objaśniających: {feature_names}')
print(f'Nazwy kategorii kosaćca: {target_names}')

# Eksploracja danych
df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['target'])
print(df.head())
plt.figure(figsize=(8, 6))
_ = sns.scatterplot('sepal_length', hue='target', data=df, legend='full',
                    palette=sns.color_palette()[:3])
print(df['target'].value_counts())

# Przygotowanie danych do modelu
data = df.copy()
data = data[['sepal_length', 'sepal_width', 'target']]
target = data.pop('target')

print(data.head())
print(target.head())

data = data.values
target = target.values.astype('int16')

# Budowa klasyfikatora drzewa decyzyjnego
#   max_depth=1
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=1, random_state=42)
classifier.fit(data, target)

#   Wykreślenie granic decyzyjnych
from mlxtend.plotting import plot_decision_regions

colors = '#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'

acc = classifier.score(data, target)

plt.figure(figsize=(8, 6))
plot_decision_regions(data, target, classifier, legend=2, colors=colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f'Drzewo decyzyjne: max_depth=1, accuracy: {acc * 100:.2f}%')
plt.show()

# Graf drzewa decyzyjnego
#   max_depth = 1
#       Wyświetlenie grafu drzewa decyzyjnego
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(classifier,
                out_file=dot_data,
                feature_names=feature_names[:2],
                class_names=target_names,
                special_characters=True,
                rounded=True,
                filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('graph.png')
Image(graph.create_png(), width=300)


# Budowa funkcji dla modelu drzewa decyzyjnego
def make_decision_tree(max_depth=1):
    # trenowanie modelu
    classifier1 = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    classifier1.fit(data, target)
    # eksport grafu drzewa
    dot_data1 = StringIO()
    export_graphviz(classifier,
                    out_file=dot_data1,
                    feature_names=feature_names[:2],
                    class_names=target_names,
                    special_characters=True,
                    rounded=True,
                    filled=True)
    graph1 = pydotplus.graph_from_dot_data(dot_data1.getvalue())
    graph1.write_png('graph.png')
    # obliczenie dokładności
    acc1 = classifier.score(data, target)
    # wykreślenie granic decyzyjnych
    colors1 = '#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'
    plt.figure(figsize=(8, 6))
    ax = plot_decision_regions(data, target, classifier, legend=0, colors=colors1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['setosa', 'versicolor', 'virginica'], framealpha=0.3)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.title(f'Drzewo decyzyjne: max_depth={max_depth}, accuracy={acc1 * 100:.2f}')

    return Image(graph.create_png(), width=200 + max_depth * 120)


make_decision_tree(max_depth=2)
make_decision_tree(max_depth=3)
make_decision_tree(max_depth=4)
make_decision_tree(max_depth=5)
make_decision_tree(max_depth=15)
