import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("olympics_dataset.csv")
df = pd.DataFrame(data)
df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.dropna(subset=['player_id', 'Year'], inplace=True)
kmeans = KMeans(n_clusters=4).fit(df[['player_id', 'Year']])
cluster_centers = kmeans.cluster_centers_
X = int(input("Enter the X-axis: "))
Y = int(input("Enter the Y-axis: "))
plt.scatter(df['player_id'], df['Year'], c=kmeans.labels_.astype(float), s=40, alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="r", s=50, marker='o')
plt.scatter(X, Y, c='k', marker='x')
plt.xlabel('Player ID')
plt.ylabel('Year')
plt.title('KMeans Clustering of Player ID vs Year')
plt.show()
