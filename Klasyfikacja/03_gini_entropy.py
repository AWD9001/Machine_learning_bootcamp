# drzewa decyzyjne

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.3)

# wskaźnik Giniego

print(1 - (50 / 150)**2 - (50 / 150)**2 - (50 / 150)**2)  # dla korzenia - poziom 0

print(1 - (45 / 52)**2 - (6 / 52)**2 - (1 / 52)**2)  # poziom 1 - węzeł lewy

print(1 - (5 / 98)**2 - (44 / 98)**2 - (49 / 98)**2)  # poziom 1 - węzeł prawy


# Entropia

print(-((50 / 150) * np.log2(50 / 150) + (50 / 150) * np.log2(50 / 150) + (50 / 150) *
        np.log2(50 / 150)))

print(-((47 / 59) * np.log2(47 / 59) + (11 / 59) * np.log2(11 / 59) + (1 / 59) * np.log2(1 / 59)))

print(-((3 / 91) * np.log2(3 / 91) + (39 / 91) * np.log2(39 / 91) + (49 / 91) * np.log2(49 / 91)))

from scipy.stats import entropy

print(entropy([0.5, 0.5], base=2))
print(entropy([0.8, 0.2], base=2))
print(entropy([0.95, 0.05], base=2))


def entropy(x):
    return -np.sum(x * np.log2(x))


print(entropy([0.5, 0.5]))
print(entropy([0.8, 0.2]))
print(entropy([0.95, 0.05]))

p = np.arange(0.01, 1.0, 0.01)
q = 1 - p
pq = np.c_[p, q]
print(pq[:10])

entropies = [entropy(pair) for pair in pq]
print(entropies[:10])

plt.plot(p, entropies)
plt.show()

entropia = -(6/10 * np.log2(6/10) + (4/10) * np.log2(4/10))
print(entropia)

entropia_wiarygodnosc = 5/10 * 0 + 2/10 * 1 + 3/10 * 0
print(entropia_wiarygodnosc)

entropia_dochod = 3/10 * 0 + 4/10 * 1 + 3/10 * 0.9183
print(entropia_dochod)

ig_wiarygodnosc = entropia - entropia_wiarygodnosc
ig_dochod = entropia - entropia_dochod

print('Zysk informacyjny (IG): wiarygodność:', ig_wiarygodnosc)
print('Zysk informacyjny (IG): dochód:', ig_dochod)
