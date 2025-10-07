import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import umap
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class ApiAnalyzer:
    """
    A class to perform a comprehensive analysis of API success factors from a given dataset.
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.feature_names = []
        self.trained_models = {}

    def prepare_data(self, target_metric="success_score"):
        print("🔧 Preparing data...")
        self.df["success_score"] = (
            self.df["views"] * 0.3
            + self.df["likes"] * 0.25
            + self.df["downloads"] * 0.25
            + self.df["rating"] * 0.2
        )
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.drop(['name', 'description', 'url', 'documentation_url', 'last_updated', 'created_date', 'scraped_at', 'endpoints', 'tags', 'features', 'sdk_languages', 'version'], errors='ignore')
        for col in categorical_cols:
            self.df[f"{col}_encoded"] = LabelEncoder().fit_transform(self.df[col].astype(str))

        self.feature_names = self.df.select_dtypes(include=np.number).columns.drop(['id', 'success_score', 'views', 'likes', 'downloads'], errors='ignore').tolist()
        self.X = self.df[self.feature_names].fillna(0)
        self.y = self.df[target_metric]
        print(f"✅ Data prepared: {len(self.df)} APIs, {len(self.feature_names)} features.")

    def perform_exploratory_analysis(self, top_n=5):
        print("\n--- Exploratory Data Analysis ---")
        print(f"\nTop {top_n} APIs by Success Score:")
        print(self.df.nlargest(top_n, "success_score")[["name", "category", "success_score"]].to_string(index=False))
        
        print(f"\nTop {top_n} Categories by Average Success Score:")
        print(self.df.groupby("category")["success_score"].mean().nlargest(top_n).to_string())

    def train_predictive_models(self):
        print("\n--- Training Predictive Models ---")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        for name, model in tqdm(models.items(), desc="Training Models"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.trained_models[name] = model
            print(f"- {name}: R² Score = {r2_score(y_test, y_pred):.3f}")

    def calculate_feature_importance(self):
        print("\n--- Calculating Feature Importance ---")
        rf_model = self.trained_models.get("Random Forest")
        if not rf_model:
            print("⚠️ Random Forest model not trained. Cannot calculate feature importance.")
            return None

        importances = pd.DataFrame({
            'Random Forest': rf_model.feature_importances_,
            'Correlation': self.X.corrwith(self.y).abs().fillna(0),
        }, index=self.feature_names)

        importances["Composite Score"] = importances.apply(lambda x: (x - x.min()) / (x.max() - x.min())).mean(axis=1) * 100
        importances = importances.sort_values("Composite Score", ascending=False)
        
        importances.to_csv("feature_importance_scores.csv")
        print("\n✅ Feature importance scores saved to 'feature_importance_scores.csv'")

        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances["Composite Score"].head(20), y=importances.head(20).index, palette="viridis")
        plt.title("Top 20 Most Important Features for API Success")
        plt.xlabel("Composite Importance Score (0-100)")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300)
        plt.close()
        print("✅ Feature importance plot saved to 'feature_importance.png'")
        return importances

    def find_optimal_clusters(self, max_k=10):
        print("\n--- Finding Optimal Number of Clusters (Elbow Method) ---")
        X_scaled = self.scaler.fit_transform(self.X)
        k_range = range(2, max_k + 1)
        inertias = [KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled).inertia_ for k in tqdm(k_range, desc="Testing k values")]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, "bo-")
        plt.xlabel("Number of Clusters (k)"); plt.ylabel("Inertia"); plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.savefig("elbow_curve.png")
        plt.close()
        print("✅ Elbow curve plot saved to 'elbow_curve.png'")

        p1, p2 = np.array([k_range[0], inertias[0]]), np.array([k_range[-1], inertias[-1]])
        distances = [np.linalg.norm(np.cross(p2 - p1, p1 - np.array([k_range[i], inertias[i]]))) / np.linalg.norm(p2 - p1) for i in range(len(k_range))]
        optimal_k = k_range[np.argmax(distances)]
        print(f"💡 Optimal number of clusters found: {optimal_k}")
        return optimal_k

    def _generate_cluster_name(self, cluster_df):
        pricing = cluster_df["pricing_model"].mode()[0]
        category = cluster_df["category"].mode()[0]
        value_desc = "Top-Rated" if cluster_df["rating"].mean() > self.df["rating"].mean() * 1.05 else "General"
        return f"{value_desc} {pricing} ({category})"

    def _plot_reduction(self, ax, data, name):
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=self.df['cluster_name'], palette='viridis', alpha=0.7, s=50, ax=ax)
        ax.set_title(f'{name} Visualization'); ax.set_xlabel(f'{name} 1'); ax.set_ylabel(f'{name} 2'); ax.legend(title='Cluster')

    def analyze_clusters(self, n_clusters=None):
        print("\n--- Clustering Analysis ---")
        n_clusters = n_clusters or self.find_optimal_clusters()
        X_scaled = self.scaler.fit_transform(self.X)
        self.df['cluster'] = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_scaled).labels_

        summaries = []
        for i in range(n_clusters):
            cluster_df = self.df[self.df['cluster'] == i]
            summaries.append({"Cluster ID": i, "Cluster Name": self._generate_cluster_name(cluster_df), "Num APIs": len(cluster_df)})
        summary_df = pd.DataFrame(summaries).set_index("Cluster Name")
        self.df['cluster_name'] = self.df['cluster'].map(summary_df.reset_index().set_index('Cluster ID')['Cluster Name'])
        print("\nCluster Profile Summary:")
        print(summary_df)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Cluster Visualization', fontsize=16)
        self._plot_reduction(axes[0], PCA(n_components=2).fit_transform(X_scaled), "PCA")
        self._plot_reduction(axes[1], umap.UMAP(random_state=42).fit_transform(X_scaled), "UMAP")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('cluster_visualizations.png', dpi=300)
        plt.close()
        print("\n✅ Cluster visualizations saved to 'cluster_visualizations.png'")

    def generate_report(self, feature_importance_df):
        print("\n--- Actionable Insights ---")
        print("\n🔥 Top 5 Critical Features:")
        print(feature_importance_df.head(5)["Composite Score"].to_string())
        print("\n📈 Top 3 Categories by Avg. Success:")
        print(self.df.groupby('category')["success_score"].mean().nlargest(3).to_string())


def main():
    print("🚀 STARTING API SUCCESS ANALYSIS")
    analyzer = ApiAnalyzer('rapidapi_data.csv')
    analyzer.prepare_data()
    analyzer.perform_exploratory_analysis()
    analyzer.train_predictive_models()
    feature_importance = analyzer.calculate_feature_importance()
    if feature_importance is not None:
        # analyzer.analyze_clusters(n_clusters=3) # <-- You can optionally override k here
        analyzer.analyze_clusters()
        analyzer.generate_report(feature_importance)
    print("\n✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()