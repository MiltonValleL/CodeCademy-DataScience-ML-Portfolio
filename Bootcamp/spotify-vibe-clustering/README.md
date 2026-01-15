# 🎵 Spotify Vibe Clustering: Unsupervised Learning & Pattern Recognition

<br>

## 👋 Overview
Welcome to my **Spotify Vibe Clustering** project! In the streaming era, user retention relies heavily on personalization and automated playlist curation. This repository features an end-to-end **Unsupervised Machine Learning** pipeline designed to categorize music tracks into distinct "vibes" (clusters) without using pre-existing genre labels.

Using a dataset of thousands of Spotify tracks, I applied **K-Means Clustering** and **Principal Component Analysis (PCA)** to group songs based on technical audio features like `danceability`, `energy`, and `acousticness`.

<br>
<br>

## 🚀 Key Highlights & Methodology
This project simulates a real-world Recommendation System task, moving from raw audio data to interpretative clusters:

---

### 1. Exploratory Data Analysis (EDA)
- **Feature Correlation**: Analyzed relationships between audio attributes (e.g., the inverse relationship between `acousticness` and `energy`) to understand the structure of the data.
- **Distribution Analysis**: Visualized the spread of data to identify skewness and potential outliers that could affect distance-based algorithms.

---

### 2. Strategic Data Preparation
- **Scaling (Normalization)**: Since K-Means is a distance-based algorithm, I utilized `StandardScaler` to normalize features. This prevents variables with larger ranges (like `loudness` or `tempo`) from dominating the Euclidean distance calculations.
- **Feature Selection**: Curated a subset of numerical features relevant to the "mood" of a song, discarding non-predictive metadata.

---

### 3. Unsupervised Modeling (The Clustering Engine)
I focused on finding the mathematical "center" of each musical vibe:
- **K-Means Clustering**: The core algorithm used to partition the dataset.
- **Hyperparameter Tuning**: Applied the **Elbow Method** (Inertia) and **Silhouette Score Analysis** to scientifically determine the optimal number of clusters ($k$), balancing cohesion with separation.
- **Dimensionality Reduction (PCA)**: Implemented Principal Component Analysis to project high-dimensional data into 2D space for visualization and to mitigate the "Curse of Dimensionality."

---

<br>
<br>

## 📊 Results & Insights

---

### The "Vibes" (Cluster Interpretation)
The model successfully identified distinct musical profiles. Based on the centroids, the clusters can be interpreted as:

> *Note: Below are examples based on the analysis. Please verify with your final notebook interpretations.*

- **Cluster 0 (e.g., The Acoustic Chill)**: High `acousticness`, low `energy`. Perfect for study or relaxation playlists.
- **Cluster 1 (e.g., High-Octane Party)**: High `danceability` and `valence`. Targeted for workout or party scenarios.
- **Cluster 2 (e.g., Lyrical/Focus)**: Distinct patterns in `speechiness` and `tempo`.

---

### Model Performance
- **Optimal k**: Determined via the Elbow method to ensure distinct, non-overlapping groups.
- **Silhouette Score**: Validated that samples are well-matched to their own cluster and poorly matched to neighboring clusters.

---

<br>
<br>

## 🛠️ Technologies Used
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Yellowbrick
- **Environment**: Jupyter Notebook / Ubuntu 24.04 LTS

---

<br>
<br>

## 📂 Repository Structure

---

```text
├── datasets/
│   └── dataset.csv                        (Spotify Tracks Data)
├── Notebooks/
│   └── Spotify_Vibe_Clustering.ipynb      (Main Clustering Pipeline)
├── README.md                              (Project documentation)
└── requirements.txt                       (Dependencies)
```

<br>
<br>

## 🔮 Strategic Recommendation (Business Perspective)
To provide Senior-level value, this project demonstrates how clustering can drive business metrics:

- **Automated Playlist Generation**: This model can be deployed to auto-generate "Mood Mixes" for users, reducing the manual labor required by human curators.

- **Cold Start Problem Solver**: For new songs with no user history, this content-based clustering allows the platform to recommend them immediately based on audio features alone.

<br>
<br>

## 🤝 Contact
I am a Data Science student on a mission to become a world-class professional by 2026. My work focuses on the intersection of statistical rigor and business impact.

If you have any questions about this project or would like to discuss Machine Learning, feel free to reach out!

**Author**: Milton Rodolfo Valle Lora

**LinkedIn:** [Please click here](https://www.linkedin.com/in/miltonvallelora/)
