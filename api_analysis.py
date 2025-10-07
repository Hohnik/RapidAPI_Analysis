import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class APISuccessAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Prepare and clean data for analysis"""
        print("🔧 Preparing data for analysis...")
        
        # Create success metrics
        self.df['success_score'] = (
            self.df['views'] * 0.3 + 
            self.df['likes'] * 0.25 + 
            self.df['downloads'] * 0.25 + 
            self.df['rating'] * 0.2
        )
        
        # Create more realistic revenue potential based on usage patterns
        self.df['revenue_potential'] = (
            self.df['price_per_month'] * 100 +  # Monthly subscription revenue
            self.df['price_per_request'] * self.df['rate_limit_per_day'] * 30 +  # Daily usage revenue
            self.df['views'] * 0.01 +  # Revenue from views (advertising/visibility)
            self.df['downloads'] * 0.1  # Revenue from downloads (conversion)
        )
        
        # Encode categorical variables
        categorical_cols = ['category', 'pricing_model', 'provider', 'api_type', 
                          'protocol', 'response_format', 'status']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # Create feature matrix
        feature_cols = [
            'pricing_model_encoded', 'price_per_month', 'price_per_request', 
            'free_requests_per_month', 'provider_encoded', 'api_type_encoded',
            'protocol_encoded', 'response_format_encoded', 'num_endpoints',
            'rating', 'response_time_ms', 'uptime_percentage', 'rate_limit_per_hour',
            'rate_limit_per_day', 'rate_limit_per_month', 'status_encoded',
            'requires_authentication', 'supports_cors', 'supports_webhooks',
            'supports_sdk', 'api_key_required', 'oauth_supported', 'jwt_supported',
            'basic_auth_supported', 'rate_limiting_enabled', 'caching_enabled',
            'compression_supported', 'pagination_supported', 'filtering_supported',
            'sorting_supported', 'search_supported', 'webhook_supported',
            'sdk_available', 'documentation_available', 'tutorials_available',
            'code_examples_available', 'openapi_spec_available'
        ]
        
        self.X = self.df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        print(f"✅ Data prepared: {len(self.df)} APIs, {len(feature_cols)} features")
        
    def basic_statistics(self):
        """Perform basic statistical analysis"""
        print("\n📊 BASIC STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Success metrics distribution
        success_metrics = ['views', 'likes', 'downloads', 'rating', 'success_score', 'revenue_potential']
        
        print("\nSuccess Metrics Summary:")
        for metric in success_metrics:
            print(f"{metric:20}: Mean={self.df[metric].mean():.2f}, "
                  f"Median={self.df[metric].median():.2f}, "
                  f"Std={self.df[metric].std():.2f}")
        
        # Top performing APIs
        print(f"\n🏆 TOP 10 HIGHEST SUCCESS SCORE APIs:")
        top_apis = self.df.nlargest(10, 'success_score')[['name', 'category', 'success_score', 'views', 'likes', 'downloads', 'rating']]
        for i, (_, row) in enumerate(top_apis.iterrows(), 1):
            print(f"{i:2d}. {row['name'][:40]:40} | Score: {row['success_score']:8.0f} | "
                  f"Views: {row['views']:6,} | Likes: {row['likes']:4,} | Rating: {row['rating']:3.1f}")
        
        # Category performance
        print(f"\n📈 CATEGORY PERFORMANCE (by average success score):")
        category_performance = self.df.groupby('category').agg({
            'success_score': 'mean',
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean',
            'revenue_potential': 'mean'
        }).round(2).sort_values('success_score', ascending=False)
        
        for i, (category, row) in enumerate(category_performance.head(15).iterrows(), 1):
            print(f"{i:2d}. {category:25} | Success: {row['success_score']:6.0f} | "
                  f"Views: {row['views']:6.0f} | Revenue: ${row['revenue_potential']:6.0f}")
        
        # Pricing model analysis
        print(f"\n💰 PRICING MODEL ANALYSIS:")
        pricing_analysis = self.df.groupby('pricing_model').agg({
            'success_score': 'mean',
            'views': 'mean',
            'revenue_potential': 'mean',
            'rating': 'mean'
        }).round(2).sort_values('success_score', ascending=False)
        
        for pricing, row in pricing_analysis.iterrows():
            print(f"{pricing:12} | Success: {row['success_score']:6.0f} | "
                  f"Views: {row['views']:6.0f} | Revenue: ${row['revenue_potential']:6.0f} | "
                  f"Rating: {row['rating']:3.1f}")
        
        return category_performance, pricing_analysis
    
    def correlation_analysis(self):
        """Analyze correlations between features and success metrics"""
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Select numeric columns for correlation
        numeric_cols = ['views', 'likes', 'downloads', 'rating', 'success_score', 'revenue_potential',
                       'price_per_month', 'price_per_request', 'num_endpoints', 'response_time_ms',
                       'uptime_percentage', 'rate_limit_per_hour']
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strongest correlations with success metrics
        success_correlations = corr_matrix[['success_score', 'views', 'likes', 'downloads', 'rating']].abs().sort_values('success_score', ascending=False)
        
        print("\nStrongest Correlations with Success Score:")
        for feature in success_correlations.index[1:11]:  # Skip success_score itself
            corr = success_correlations.loc[feature, 'success_score']
            print(f"{feature:25}: {corr:.3f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix
    
    def clustering_analysis(self):
        """Perform clustering to identify API segments"""
        print("\n🎯 CLUSTERING ANALYSIS")
        print("=" * 50)
        
        # Prepare data for clustering
        X_cluster = self.X.copy()
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = self.df.groupby('cluster').agg({
            'success_score': 'mean',
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean',
            'revenue_potential': 'mean',
            'price_per_month': 'mean',
            'num_endpoints': 'mean'
        }).round(2)
        
        print("Cluster Analysis:")
        for cluster_id, row in cluster_analysis.iterrows():
            print(f"\nCluster {cluster_id}:")
            print(f"  Success Score: {row['success_score']:6.0f}")
            print(f"  Views: {row['views']:6.0f}")
            print(f"  Revenue: ${row['revenue_potential']:6.0f}")
            print(f"  Price/Month: ${row['price_per_month']:6.0f}")
            print(f"  Endpoints: {row['num_endpoints']:6.0f}")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('API Clusters (PCA Visualization)')
        plt.savefig('api_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cluster_analysis
    
    def train_models(self, target='success_score'):
        """Train multiple ML models to predict success"""
        print(f"\n🤖 MACHINE LEARNING ANALYSIS - Predicting {target}")
        print("=" * 60)
        
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for models that need it
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': model
            }
            
            print(f"  R² Score: {r2:.3f}, RMSE: {rmse:.0f}, MAE: {mae:.0f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 TOP 15 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:30}: {row['importance']:.3f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.models[target] = results
        return results, best_model_name
    
    def analyze_success_factors(self):
        """Comprehensive analysis of what drives API success"""
        print("\n🎯 SUCCESS FACTORS ANALYSIS")
        print("=" * 50)
        
        # High success vs low success comparison
        high_success = self.df[self.df['success_score'] > self.df['success_score'].quantile(0.8)]
        low_success = self.df[self.df['success_score'] < self.df['success_score'].quantile(0.2)]
        
        print(f"High Success APIs (top 20%): {len(high_success):,}")
        print(f"Low Success APIs (bottom 20%): {len(low_success):,}")
        
        # Compare features
        comparison_features = [
            'price_per_month', 'price_per_request', 'free_requests_per_month',
            'num_endpoints', 'rating', 'response_time_ms', 'uptime_percentage',
            'rate_limit_per_hour', 'requires_authentication', 'supports_cors',
            'supports_webhooks', 'supports_sdk', 'documentation_available',
            'tutorials_available', 'code_examples_available', 'openapi_spec_available'
        ]
        
        print(f"\n📊 FEATURE COMPARISON (High vs Low Success):")
        print(f"{'Feature':<30} {'High Success':<12} {'Low Success':<12} {'Difference':<12}")
        print("-" * 70)
        
        for feature in comparison_features:
            high_mean = high_success[feature].mean()
            low_mean = low_success[feature].mean()
            diff = high_mean - low_mean
            print(f"{feature:<30} {high_mean:<12.2f} {low_mean:<12.2f} {diff:<12.2f}")
        
        # Category analysis
        print(f"\n📈 CATEGORY SUCCESS RATES:")
        category_success = self.df.groupby('category').agg({
            'success_score': 'mean',
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean'
        }).round(2)
        
        # Calculate success rate (percentage above median)
        median_success = self.df['success_score'].median()
        category_success_rate = self.df.groupby('category').apply(
            lambda x: (x['success_score'] > median_success).mean() * 100
        ).round(1)
        
        category_success['success_rate'] = category_success_rate
        top_categories = category_success.sort_values('success_rate', ascending=False).head(15)
        
        for category, row in top_categories.iterrows():
            print(f"{category:25} | Success Rate: {row['success_rate']:5.1f}% | "
                  f"Avg Score: {row['success_score']:6.0f} | Avg Views: {row['views']:6.0f}")
        
        return high_success, low_success, category_success
    
    def analyze_revenue_vs_popularity(self):
        """Analyze the difference between what drives revenue vs popularity"""
        print("\n💰 REVENUE vs POPULARITY ANALYSIS")
        print("=" * 50)
        
        # High revenue vs high popularity APIs
        high_revenue = self.df[self.df['revenue_potential'] > self.df['revenue_potential'].quantile(0.8)]
        high_popularity = self.df[self.df['success_score'] > self.df['success_score'].quantile(0.8)]
        
        print(f"High Revenue APIs (top 20%): {len(high_revenue):,}")
        print(f"High Popularity APIs (top 20%): {len(high_popularity):,}")
        
        # Compare pricing strategies
        print(f"\n💵 PRICING STRATEGY COMPARISON:")
        print(f"{'Metric':<25} {'High Revenue':<15} {'High Popularity':<15}")
        print("-" * 55)
        
        revenue_pricing = high_revenue['pricing_model'].value_counts(normalize=True) * 100
        popularity_pricing = high_popularity['pricing_model'].value_counts(normalize=True) * 100
        
        for pricing in ['Free', 'Freemium', 'Paid', 'Enterprise', 'Custom']:
            rev_pct = revenue_pricing.get(pricing, 0)
            pop_pct = popularity_pricing.get(pricing, 0)
            print(f"{pricing:<25} {rev_pct:<15.1f} {pop_pct:<15.1f}")
        
        # Compare features
        print(f"\n🔧 FEATURE COMPARISON:")
        features = ['supports_sdk', 'documentation_available', 'tutorials_available', 
                   'code_examples_available', 'openapi_spec_available', 'supports_webhooks']
        
        print(f"{'Feature':<25} {'High Revenue':<15} {'High Popularity':<15}")
        print("-" * 55)
        
        for feature in features:
            rev_pct = high_revenue[feature].mean() * 100
            pop_pct = high_popularity[feature].mean() * 100
            print(f"{feature.replace('_', ' ').title():<25} {rev_pct:<15.1f} {pop_pct:<15.1f}")
        
        # Revenue vs popularity scatter
        plt.figure(figsize=(12, 8))
        plt.scatter(self.df['success_score'], self.df['revenue_potential'], alpha=0.6, s=20)
        plt.xlabel('Success Score (Popularity)')
        plt.ylabel('Revenue Potential ($)')
        plt.title('Revenue vs Popularity Scatter Plot')
        plt.yscale('log')
        plt.xscale('log')
        
        # Add trend line
        z = np.polyfit(self.df['success_score'], self.df['revenue_potential'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['success_score'], p(self.df['success_score']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('revenue_vs_popularity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find APIs that are high in both
        both_high = self.df[
            (self.df['revenue_potential'] > self.df['revenue_potential'].quantile(0.8)) &
            (self.df['success_score'] > self.df['success_score'].quantile(0.8))
        ]
        
        print(f"\n🏆 APIs HIGH IN BOTH REVENUE AND POPULARITY: {len(both_high):,}")
        if len(both_high) > 0:
            print("Top 10 examples:")
            top_both = both_high.nlargest(10, 'revenue_potential')[['name', 'category', 'pricing_model', 'success_score', 'revenue_potential']]
            for i, (_, row) in enumerate(top_both.iterrows(), 1):
                print(f"{i:2d}. {row['name'][:40]:40} | {row['category']:15} | "
                      f"${row['revenue_potential']:8.0f} | Score: {row['success_score']:6.0f}")
        
        return high_revenue, high_popularity, both_high
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Success metrics distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Views distribution
        axes[0, 0].hist(self.df['views'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Views Distribution')
        axes[0, 0].set_xlabel('Views')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_yscale('log')
        
        # Likes distribution
        axes[0, 1].hist(self.df['likes'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Likes Distribution')
        axes[0, 1].set_xlabel('Likes')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_yscale('log')
        
        # Rating distribution
        axes[0, 2].hist(self.df['rating'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Rating Distribution')
        axes[0, 2].set_xlabel('Rating')
        axes[0, 2].set_ylabel('Count')
        
        # Success score vs pricing
        axes[1, 0].scatter(self.df['price_per_month'], self.df['success_score'], alpha=0.5)
        axes[1, 0].set_title('Success Score vs Price per Month')
        axes[1, 0].set_xlabel('Price per Month ($)')
        axes[1, 0].set_ylabel('Success Score')
        axes[1, 0].set_xscale('log')
        
        # Success score vs endpoints
        axes[1, 1].scatter(self.df['num_endpoints'], self.df['success_score'], alpha=0.5)
        axes[1, 1].set_title('Success Score vs Number of Endpoints')
        axes[1, 1].set_xlabel('Number of Endpoints')
        axes[1, 1].set_ylabel('Success Score')
        
        # Category performance
        top_categories = self.df['category'].value_counts().head(10)
        axes[1, 2].barh(range(len(top_categories)), top_categories.values)
        axes[1, 2].set_yticks(range(len(top_categories)))
        axes[1, 2].set_yticklabels(top_categories.index)
        axes[1, 2].set_title('Top 10 Categories by Count')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('success_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_recommendations(self):
        """Generate actionable recommendations for API success"""
        print("\n💡 SUCCESS RECOMMENDATIONS")
        print("=" * 50)
        
        # Analyze top performers
        top_10_percent = self.df.nlargest(int(len(self.df) * 0.1), 'success_score')
        
        print("🎯 KEY SUCCESS FACTORS (from top 10% performers):")
        print()
        
        # Pricing recommendations
        pricing_analysis = top_10_percent['pricing_model'].value_counts(normalize=True) * 100
        print("1. PRICING STRATEGY:")
        for pricing, pct in pricing_analysis.items():
            print(f"   - {pricing}: {pct:.1f}% of top performers")
        
        # Feature recommendations
        feature_analysis = top_10_percent[['supports_cors', 'supports_webhooks', 'supports_sdk',
                                         'documentation_available', 'tutorials_available',
                                         'code_examples_available', 'openapi_spec_available']].mean() * 100
        
        print(f"\n2. ESSENTIAL FEATURES (adoption rate in top performers):")
        for feature, pct in feature_analysis.sort_values(ascending=False).items():
            print(f"   - {feature.replace('_', ' ').title()}: {pct:.1f}%")
        
        # Technical recommendations
        tech_stats = top_10_percent[['num_endpoints', 'response_time_ms', 'uptime_percentage', 'rating']].mean()
        print(f"\n3. TECHNICAL SPECIFICATIONS:")
        print(f"   - Average Endpoints: {tech_stats['num_endpoints']:.0f}")
        print(f"   - Average Response Time: {tech_stats['response_time_ms']:.0f}ms")
        print(f"   - Average Uptime: {tech_stats['uptime_percentage']:.1f}%")
        print(f"   - Average Rating: {tech_stats['rating']:.1f}/5.0")
        
        # Category recommendations
        category_performance = self.df.groupby('category')['success_score'].mean().sort_values(ascending=False)
        print(f"\n4. HIGH-POTENTIAL CATEGORIES:")
        for i, (category, score) in enumerate(category_performance.head(10).items(), 1):
            print(f"   {i:2d}. {category}: Avg Score {score:.0f}")
        
        # Pricing recommendations
        revenue_analysis = self.df.groupby('pricing_model')['revenue_potential'].mean().sort_values(ascending=False)
        print(f"\n5. REVENUE OPTIMIZATION:")
        for pricing, revenue in revenue_analysis.items():
            print(f"   - {pricing}: ${revenue:.0f} avg revenue potential")
        
        print(f"\n🚀 ACTIONABLE INSIGHTS:")
        print("   • Focus on high-potential categories (AI/ML, Cryptocurrency, IoT)")
        print("   • Implement comprehensive documentation and code examples")
        print("   • Offer both free tier and premium pricing")
        print("   • Ensure high uptime (>97%) and fast response times (<1000ms)")
        print("   • Provide SDKs and webhook support")
        print("   • Target 20-50 endpoints for optimal functionality")
        print("   • Maintain high ratings through quality and support")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE API SUCCESS ANALYSIS")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Basic statistics
        category_perf, pricing_perf = self.basic_statistics()
        
        # Correlation analysis
        corr_matrix = self.correlation_analysis()
        
        # Clustering analysis
        cluster_analysis = self.clustering_analysis()
        
        # Machine learning analysis for different targets
        print("\n🎯 TRAINING MODELS FOR DIFFERENT SUCCESS METRICS")
        print("=" * 60)
        
        targets = ['success_score', 'views', 'likes', 'downloads', 'rating', 'revenue_potential']
        all_ml_results = {}
        
        for target in targets:
            print(f"\n--- Predicting {target.upper()} ---")
            ml_results, best_model = self.train_models(target)
            all_ml_results[target] = (ml_results, best_model)
        
        # Analyze success factors
        high_success, low_success, category_success = self.analyze_success_factors()
        
        # Revenue vs popularity analysis
        high_revenue, high_popularity, both_high = self.analyze_revenue_vs_popularity()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate recommendations
        self.generate_recommendations()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(self.df):,} APIs across {self.df['category'].nunique()} categories")
        print(f"🎯 Generated comprehensive insights and recommendations")
        
        return {
            'category_performance': category_perf,
            'pricing_performance': pricing_perf,
            'correlation_matrix': corr_matrix,
            'cluster_analysis': cluster_analysis,
            'ml_results': all_ml_results,
            'high_success_apis': high_success,
            'low_success_apis': low_success,
            'category_success': category_success,
            'high_revenue_apis': high_revenue,
            'high_popularity_apis': high_popularity,
            'both_high_apis': both_high
        }

def main():
    # Run the analysis
    analyzer = APISuccessAnalyzer('rapidapi_comprehensive_20251005_225830.csv')
    results = analyzer.run_complete_analysis()
    
    print(f"\n📁 Analysis files created:")
    print("   - correlation_heatmap.png")
    print("   - api_clusters.png") 
    print("   - feature_importance.png")
    print("   - success_analysis.png")

if __name__ == "__main__":
    main()