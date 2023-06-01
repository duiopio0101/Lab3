import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Функція для генерації тестової послідовності на одиничному квадраті
def generate_data(n):
    return np.random.rand(n, 2)

# Функція для обчислення міри віддалі (в даному випадку Евклідова відстань)
def distance_measure(x, y):
    return np.linalg.norm(x - y)

# Функція для виконання кластеризації за методом К-середніх
def kmeans_clustering(data, k):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return labels, cluster_centers

# Функція для виконання кластеризації за ієрархічним методом
def hierarchical_clustering(data, k):
    clustering = AgglomerativeClustering(n_clusters=k)
    labels = clustering.fit_predict(data)
    return labels

# Функція для обчислення середньо-зваженого розміру утворених кластерів за мірою віддалі
def compute_cluster_sizes(data, labels):
    sizes = []
    for cluster_label in np.unique(labels):
        cluster_points = data[labels == cluster_label]
        cluster_size = len(cluster_points)
        sizes.append(cluster_size)
    return sizes

# Функція для візуалізації результатів кластеризації
def plot_clusters(data, labels, centers=None):
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clustering Results')
    plt.show()

# Генерування тестової послідовності з 1000 значень
data = generate_data(2000)

# Виконання кластеризації за методом К-середніх
k = 5 # кількість кластерів
k1 = 5
kmeans_labels, kmeans_centers = kmeans_clustering(data, k)

# Визуалізація результатів кластеризації за методом К-середніх
plot_clusters(data, kmeans_labels, kmeans_centers)

# Виконання кластеризації за ієрархічним методом
hierarchical_labels = hierarchical_clustering(data, k1)

# Визуалізація результатів кластеризації за ієрархічним методом
plot_clusters(data, hierarchical_labels)

# Обчислення середньо-зваженого розміру утворених кластерів за мірою віддалі
kmeans_cluster_sizes = compute_cluster_sizes(data, kmeans_labels)
hierarchical_cluster_sizes = compute_cluster_sizes(data, hierarchical_labels)

# Порівняння результатів кластеризації
num_kmeans_clusters = len(np.unique(kmeans_labels))
num_hierarchical_clusters = len(np.unique(hierarchical_labels))
mean_kmeans_cluster_size = np.mean(kmeans_cluster_sizes)
mean_hierarchical_cluster_size = np.mean(hierarchical_cluster_sizes)

print("Кількість кластерів за методом К-середніх:", num_kmeans_clusters)
print("Кількість кластерів за ієрархічним методом:", num_hierarchical_clusters)
print("Середньо-зважені розміри кластерів за методом К-середніх:", mean_kmeans_cluster_size)
print("Середньо-зважені розміри кластерів за ієрархічним методом:", mean_hierarchical_cluster_size)
