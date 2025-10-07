import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class FocusedAPIAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.selected_features = []
        
    def prepare_data(self):
        """Prepare data and create success score"""
        print("🔧 Preparing data for focused analysis...")
        
        # Create success score (most important target)
        self.df['success_score'] = (
            self.df['views'] * 0.3 + 
            self.df['likes'] * 0.25 + 
            self.df['downloads'] * 0.25 + 
            self.df['rating'] * 0.2
        )
        
        # Encode categorical variables
        categorical_cols = ['category', 'pricing_model', 'provider', 'api_type', 
                          'protocol', 'response_format', 'status']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"✅ Data prepared: {len(self.df)} APIs")
        
    def identify_zero_correlation_features(self):
        """Identify and remove features with zero correlation to success_score"""
        print("\n🎯 IDENTIFYING ZERO-CORRELATION FEATURES")
        print("=" * 50)
        
        # Get all numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f not in ['id', 'success_score', 'views', 'likes', 'downloads', 'rating']]
        
        # Calculate correlations with success_score
        correlations = []
        for feature in numeric_features:
            corr = self.df[feature].corr(self.df['success_score'])
            correlations.append((feature, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Feature correlations with success_score:")
        print(f"{'Feature':<40} {'Correlation':<12} {'Status'}")
        print("-" * 65)
        
        zero_corr_features = []
        weak_corr_features = []
        
        for feature, corr in correlations:
            if abs(corr) < 0.01:
                status = "❌ ZERO"
                zero_corr_features.append(feature)
            elif abs(corr) < 0.1:
                status = "⚠️  WEAK"
                weak_corr_features.append(feature)
            else:
                status = "✅ GOOD"
            
            print(f"{feature:<40} {corr:<12.4f} {status}")
        
        print(f"\n📊 SUMMARY:")
        print(f"Zero correlation features: {len(zero_corr_features)}")
        print(f"Weak correlation features: {len(weak_corr_features)}")
        print(f"Good correlation features: {len(correlations) - len(zero_corr_features) - len(weak_corr_features)}")
        
        # Remove zero correlation features
        self.df_clean = self.df.drop(columns=zero_corr_features)
        print(f"\n🧹 Removed {len(zero_corr_features)} zero-correlation features")
        
        return zero_corr_features, weak_corr_features, correlations
    
    def feature_selection(self):
        """Use multiple methods to select the most important features"""
        print("\n🔍 FEATURE SELECTION ANALYSIS")
        print("=" * 50)
        
        # Prepare feature matrix (exclude zero correlation features and non-numeric columns)
        exclude_cols = ['id', 'name', 'description', 'category', 'pricing_model', 
                       'provider', 'api_type', 'protocol', 'response_format', 'status',
                       'url', 'documentation_url', 'last_updated', 'created_date',
                       'scraped_at', 'success_score', 'views', 'likes', 'downloads', 'rating',
                       'endpoints', 'tags', 'features', 'sdk_languages', 'version']
        
        feature_cols = [col for col in self.df_clean.columns if col not in exclude_cols]
        
        # Only use numeric columns
        numeric_cols = self.df_clean[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = self.df_clean[numeric_cols].fillna(0)
        y = self.df_clean['success_score']
        
        print(f"Starting with {len(feature_cols)} features")
        
        # Method 1: Correlation-based selection
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        high_corr_features = correlations[correlations > 0.1].index.tolist()
        print(f"High correlation features (>0.1): {len(high_corr_features)}")
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_features = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores})
        mi_features = mi_features.sort_values('mi_score', ascending=False)
        top_mi_features = mi_features.head(20)['feature'].tolist()
        print(f"Top 20 mutual information features: {len(top_mi_features)}")
        
        # Method 3: Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        top_rf_features = rf_importance.head(20)['feature'].tolist()
        print(f"Top 20 Random Forest features: {len(top_rf_features)}")
        
        # Method 4: SelectKBest
        selector = SelectKBest(score_func=f_regression, k=20)
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        print(f"SelectKBest top 20 features: {len(selected_features)}")
        
        # Combine all methods
        all_important = set(high_corr_features) | set(top_mi_features) | set(top_rf_features) | set(selected_features)
        print(f"\nCombined important features: {len(all_important)}")
        
        # Final feature selection based on consensus
        feature_votes = {}
        for feature in all_important:
            votes = 0
            if feature in high_corr_features:
                votes += 1
            if feature in top_mi_features:
                votes += 1
            if feature in top_rf_features:
                votes += 1
            if feature in selected_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes
        self.selected_features = [f for f, votes in feature_votes.items() if votes >= 2]
        print(f"Final selected features (≥2 votes): {len(self.selected_features)}")
        
        # Show top features
        print(f"\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
        for i, feature in enumerate(self.selected_features[:15], 1):
            votes = feature_votes[feature]
            corr = correlations.get(feature, 0)
            print(f"{i:2d}. {feature:<35} | Votes: {votes} | Corr: {corr:.3f}")
        
        return self.selected_features, feature_votes, correlations
    
    def analyze_clusters(self):
        """Analyze the distinct API clusters"""
        print("\n🎯 CLUSTER ANALYSIS")
        print("=" * 50)
        
        # Prepare data for clustering
        X_cluster = self.df_clean[self.selected_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal Clusters')
        plt.grid(True)
        plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Use 5 clusters (based on typical API market segments)
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df_clean['cluster'] = clusters
        
        # Analyze each cluster
        print(f"\n📊 CLUSTER ANALYSIS ({n_clusters} clusters):")
        cluster_analysis = self.df_clean.groupby('cluster').agg({
            'success_score': ['mean', 'std', 'count'],
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean',
            'price_per_month': 'mean',
            'num_endpoints': 'mean',
            'response_time_ms': 'mean',
            'uptime_percentage': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns]
        
        print(f"\n{'Cluster':<8} {'Count':<8} {'Success':<10} {'Views':<10} {'Rating':<8} {'Price':<10} {'Endpoints':<10}")
        print("-" * 80)
        
        for cluster_id in range(n_clusters):
            row = cluster_analysis.loc[cluster_id]
            print(f"{cluster_id:<8} {row['success_score_count']:<8.0f} {row['success_score_mean']:<10.0f} "
                  f"{row['views_mean']:<10.0f} {row['rating_mean']:<8.2f} {row['price_per_month_mean']:<10.0f} "
                  f"{row['num_endpoints_mean']:<10.0f}")
        
        # Analyze cluster characteristics
        print(f"\n🔍 CLUSTER CHARACTERISTICS:")
        for cluster_id in range(n_clusters):
            cluster_data = self.df_clean[self.df_clean['cluster'] == cluster_id]
            print(f"\n--- CLUSTER {cluster_id} ---")
            print(f"Size: {len(cluster_data):,} APIs ({len(cluster_data)/len(self.df_clean)*100:.1f}%)")
            print(f"Avg Success Score: {cluster_data['success_score'].mean():.0f}")
            print(f"Top Categories: {cluster_data['category'].value_counts().head(3).to_dict()}")
            print(f"Top Pricing: {cluster_data['pricing_model'].value_counts().head(2).to_dict()}")
            
            # Key features of this cluster
            key_features = []
            for feature in self.selected_features[:10]:
                if cluster_data[feature].mean() > self.df_clean[feature].mean() * 1.2:
                    key_features.append(feature)
            print(f"Key Features: {key_features[:5]}")
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('API Clusters (PCA Visualization)')
        plt.savefig('api_clusters_focused.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cluster_analysis, clusters
    
    def train_focused_models(self):
        """Train models using only the selected important features"""
        print("\n🤖 FOCUSED MODEL TRAINING")
        print("=" * 50)
        
        X = self.df_clean[self.selected_features].fillna(0)
        y = self.df_clean['success_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Neural Network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {'mse': mse, 'rmse': rmse, 'r2': r2, 'model': model}
            print(f"  R² Score: {r2:.3f}, RMSE: {rmse:.0f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 FEATURE IMPORTANCE (Top 10):")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<35}: {row['importance']:.3f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance_focused.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return results, best_model_name
    
    def cluster_success_analysis(self):
        """Analyze what makes each cluster successful"""
        print("\n🎯 CLUSTER SUCCESS ANALYSIS")
        print("=" * 50)
        
        # Analyze each cluster's success patterns
        for cluster_id in range(5):
            cluster_data = self.df_clean[self.df_clean['cluster'] == cluster_id]
            
            print(f"\n--- CLUSTER {cluster_id} SUCCESS ANALYSIS ---")
            print(f"Size: {len(cluster_data):,} APIs")
            print(f"Success Score: {cluster_data['success_score'].mean():.0f} ± {cluster_data['success_score'].std():.0f}")
            
            # Top performing APIs in this cluster
            top_in_cluster = cluster_data.nlargest(5, 'success_score')[['name', 'category', 'success_score', 'views', 'likes', 'rating']]
            print(f"Top 5 APIs in cluster:")
            for i, (_, row) in enumerate(top_in_cluster.iterrows(), 1):
                print(f"  {i}. {row['name'][:40]:40} | Score: {row['success_score']:6.0f} | "
                      f"Views: {row['views']:6,} | Rating: {row['rating']:3.1f}")
            
            # Key success factors for this cluster
            print(f"Key success factors:")
            for feature in self.selected_features[:10]:
                high_success = cluster_data[cluster_data['success_score'] > cluster_data['success_score'].quantile(0.8)]
                low_success = cluster_data[cluster_data['success_score'] < cluster_data['success_score'].quantile(0.2)]
                
                if len(high_success) > 0 and len(low_success) > 0:
                    high_mean = high_success[feature].mean()
                    low_mean = low_success[feature].mean()
                    if abs(high_mean - low_mean) > cluster_data[feature].std() * 0.5:
                        print(f"  - {feature}: High={high_mean:.2f}, Low={low_mean:.2f}")
    
    def generate_focused_recommendations(self):
        """Generate focused recommendations based on cluster analysis"""
        print("\n💡 FOCUSED SUCCESS RECOMMENDATIONS")
        print("=" * 50)
        
        # Analyze the most successful cluster
        cluster_success = self.df_clean.groupby('cluster')['success_score'].mean().sort_values(ascending=False)
        best_cluster = cluster_success.index[0]
        best_cluster_data = self.df_clean[self.df_clean['cluster'] == best_cluster]
        
        print(f"🏆 MOST SUCCESSFUL CLUSTER: {best_cluster}")
        print(f"Average Success Score: {cluster_success[best_cluster]:.0f}")
        print(f"Size: {len(best_cluster_data):,} APIs")
        
        # Key characteristics of successful cluster
        print(f"\nKey Characteristics of Successful APIs:")
        
        # Categorical features
        print(f"Top Categories: {best_cluster_data['category'].value_counts().head(3).to_dict()}")
        print(f"Pricing Models: {best_cluster_data['pricing_model'].value_counts().head(3).to_dict()}")
        print(f"API Types: {best_cluster_data['api_type'].value_counts().head(3).to_dict()}")
        
        # Numeric features
        for feature in self.selected_features[:10]:
            if feature in best_cluster_data.columns:
                mean_val = best_cluster_data[feature].mean()
                overall_mean = self.df_clean[feature].mean()
                if abs(mean_val - overall_mean) > overall_mean * 0.1:
                    print(f"{feature}: {mean_val:.2f} (vs {overall_mean:.2f} overall)")
        
        # Actionable recommendations
        print(f"\n🚀 ACTIONABLE RECOMMENDATIONS:")
        print(f"1. Focus on categories: {best_cluster_data['category'].value_counts().head(3).index.tolist()}")
        print(f"2. Use pricing model: {best_cluster_data['pricing_model'].mode().iloc[0]}")
        print(f"3. Target API type: {best_cluster_data['api_type'].mode().iloc[0]}")
        
        # Feature recommendations
        high_features = []
        for feature in self.selected_features:
            if feature in best_cluster_data.columns:
                if best_cluster_data[feature].mean() > self.df_clean[feature].mean() * 1.2:
                    high_features.append(feature)
        
        print(f"4. Key features to implement: {high_features[:5]}")
        
        # Avoid characteristics of worst cluster
        worst_cluster = cluster_success.index[-1]
        worst_cluster_data = self.df_clean[self.df_clean['cluster'] == worst_cluster]
        
        print(f"\n⚠️  AVOID (worst cluster characteristics):")
        print(f"Categories to avoid: {worst_cluster_data['category'].value_counts().head(3).index.tolist()}")
        print(f"Pricing to avoid: {worst_cluster_data['pricing_model'].mode().iloc[0]}")
    
    def run_focused_analysis(self):
        """Run the complete focused analysis"""
        print("🎯 FOCUSED API SUCCESS ANALYSIS")
        print("=" * 60)
        print("Focusing on features that actually matter for success_score")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Identify and remove zero correlation features
        zero_features, weak_features, correlations = self.identify_zero_correlation_features()
        
        # Feature selection
        selected_features, feature_votes, correlations = self.feature_selection()
        
        # Cluster analysis
        cluster_analysis, clusters = self.analyze_clusters()
        
        # Train focused models
        model_results, best_model = self.train_focused_models()
        
        # Cluster success analysis
        self.cluster_success_analysis()
        
        # Generate focused recommendations
        self.generate_focused_recommendations()
        
        print(f"\n✅ FOCUSED ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(self.df):,} APIs")
        print(f"🎯 Identified {len(selected_features)} truly important features")
        print(f"🔍 Found {len(set(clusters))} distinct API clusters")
        print(f"🏆 Best model: {best_model} with focused features")
        
        return {
            'selected_features': selected_features,
            'zero_correlation_features': zero_features,
            'cluster_analysis': cluster_analysis,
            'model_results': model_results,
            'best_model': best_model,
            'correlations': correlations
        }

def main():
    analyzer = FocusedAPIAnalyzer('rapidapi_comprehensive_20251005_225830.csv')
    results = analyzer.run_focused_analysis()
    
    print(f"\n📁 Focused analysis files created:")
    print("   - elbow_curve.png")
    print("   - api_clusters_focused.png")
    print("   - feature_importance_focused.png")

if __name__ == "__main__":
    main()