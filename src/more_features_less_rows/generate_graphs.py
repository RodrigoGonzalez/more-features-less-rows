import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.more_features_less_rows.feature_selection import data_transporter

sns.set_theme(style="whitegrid")


def generate_scatterplot():
    data_location = "data/anonymized_data.csv"
    dt = data_transporter(data_location)
    X, y = dt.X, dt.y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA with all components
    pca_all = PCA(n_components=None, random_state=1234)
    pca_all.fit(X_scaled)

    # Compute the explained variance ratio and cumulative explained variance
    explained_variance_ratio_all = pca_all.explained_variance_ratio_
    cumulative_explained_variance_all = explained_variance_ratio_all.cumsum()

    # Create a DataFrame for the scatter plot
    variance_df = pd.DataFrame(
        {
            "Number of Components": range(1, len(cumulative_explained_variance_all) + 1),
            "Cumulative Explained Variance": cumulative_explained_variance_all,
            "Target": y,
            "Individual Variance": explained_variance_ratio_all * 5000,
            # Scale for better visualization
            "Explained Variance": explained_variance_ratio_all,
        }
    )

    # Create the scatter plot using Seaborn
    plt.figure(figsize=(10, 6))

    # Get the current axes
    ax = plt.gca()

    # Set the face color of the plot area to light grey
    ax.set_facecolor("#f2f2f2")

    sns.scatterplot(
        x="Number of Components",
        y="Cumulative Explained Variance",
        hue="Explained Variance",
        size="Individual Variance",
        sizes=(50, 250),  # Range of sizes
        palette="viridis",
        data=variance_df,
    )

    plt.title("Cumulative Explained Variance by Number of Components", fontsize=20)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend(loc="lower right", labelspacing=1, fancybox=True, shadow=True, markerscale=1.2)
    # plt.show()

    plt.savefig("assets/pca_plot.png", dpi=300, bbox_inches="tight")

    # # Apply PCA with all components
    # pca_all = PCA(n_components=None)
    # principal_components_all = pca_all.fit_transform(X_scaled)
    #
    # # Compute the explained variance ratio and cumulative explained variance
    # explained_variance_ratio_all = pca_all.explained_variance_ratio_
    # cumulative_explained_variance_all = explained_variance_ratio_all.cumsum()
    #
    # # Create a DataFrame for the first two principal components and cumulative explained variance
    # scatter_df = pd.DataFrame({
    # 	'principal component 1': principal_components_all[:, 0],
    # 	'principal component 2': principal_components_all[:, 1],
    # 	'cumulative explained variance': cumulative_explained_variance_all
    # })
    #
    # # Scale the cumulative explained variance to a suitable size range
    # scatter_df['size'] = (scatter_df['cumulative explained variance'] * 500) + 50
    #
    # # Create the scatter plot using the size for the cumulative explained variance
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(
    # 	x='principal component 1',
    # 	y='principal component 2',
    # 	palette="ch:r=-.2,d=.3_r",
    # 	hue_order='cumulative explained variance',
    # 	hue='cumulative explained variance',
    # 	size='size',
    # 	sizes=(50, 500),
    # 	data=scatter_df,
    # 	legend=False
    # )
    # plt.title('PCA Scatter Plot with Cumulative Explained Variance')
    # plt.show()
