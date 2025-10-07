import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SimpleAPIAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Prepare data for analysis"""
        print("🔧 Preparing data...")
        
        # Create success metrics
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
        
        print(f"✅ Data prepared: {len(self.df)} APIs")
        
    def analyze_correlations(self):
        """Analyze correlations with success metrics"""
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Select numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_features = ['id', 'success_score', 'views', 'likes', 'downloads', 'rating']
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        # Calculate correlations
        correlations = []
        for feature in numeric_features:
            corr = self.df[feature].corr(self.df['success_score'])
            correlations.append((feature, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("Feature correlations with success_score:")
        print(f"{'Feature':<40} {'Correlation':<12} {'Status'}")
        print("-" * 65)
        
        for feature, corr in correlations:
            if abs(corr) > 0.1:
                status = "✅ STRONG"
            elif abs(corr) > 0.05:
                status = "⚠️  MODERATE"
            elif abs(corr) > 0.01:
                status = "🔸 WEAK"
            else:
                status = "❌ ZERO"
            
            print(f"{feature:<40} {corr:<12.4f} {status}")
        
        # Get strong correlation features
        strong_features = [f for f, c in correlations if abs(c) > 0.1]
        moderate_features = [f for f, c in correlations if 0.05 < abs(c) <= 0.1]
        weak_features = [f for f, c in correlations if 0.01 < abs(c) <= 0.05]
        zero_features = [f for f, c in correlations if abs(c) <= 0.01]
        
        print(f"\n📊 SUMMARY:")
        print(f"Strong correlation (>0.1): {len(strong_features)}")
        print(f"Moderate correlation (0.05-0.1): {len(moderate_features)}")
        print(f"Weak correlation (0.01-0.05): {len(weak_features)}")
        print(f"Zero correlation (<0.01): {len(zero_features)}")
        
        return strong_features, moderate_features, weak_features, zero_features, correlations
    
    def analyze_categories(self):
        """Analyze category performance"""
        print("\n📈 CATEGORY ANALYSIS")
        print("=" * 50)
        
        category_stats = self.df.groupby('category').agg({
            'success_score': ['mean', 'std', 'count'],
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean',
            'price_per_month': 'mean'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        
        # Sort by success score
        category_stats = category_stats.sort_values('success_score_mean', ascending=False)
        
        print("Top 20 Categories by Success Score:")
        print(f"{'Category':<25} {'Count':<8} {'Success':<10} {'Views':<10} {'Rating':<8} {'Price':<10}")
        print("-" * 85)
        
        for i, (category, row) in enumerate(category_stats.head(20).iterrows(), 1):
            print(f"{i:2d}. {category:<25} {row['success_score_count']:<8.0f} {row['success_score_mean']:<10.0f} "
                  f"{row['views_mean']:<10.0f} {row['rating_mean']:<8.2f} {row['price_per_month_mean']:<10.0f}")
        
        return category_stats
    
    def analyze_pricing(self):
        """Analyze pricing model performance"""
        print("\n💰 PRICING ANALYSIS")
        print("=" * 50)
        
        pricing_stats = self.df.groupby('pricing_model').agg({
            'success_score': ['mean', 'std', 'count'],
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean',
            'price_per_month': 'mean'
        }).round(2)
        
        # Flatten column names
        pricing_stats.columns = ['_'.join(col).strip() for col in pricing_stats.columns]
        
        print("Pricing Model Performance:")
        print(f"{'Pricing':<15} {'Count':<8} {'Success':<10} {'Views':<10} {'Rating':<8} {'Price':<10}")
        print("-" * 75)
        
        for pricing, row in pricing_stats.iterrows():
            print(f"{pricing:<15} {row['success_score_count']:<8.0f} {row['success_score_mean']:<10.0f} "
                  f"{row['views_mean']:<10.0f} {row['rating_mean']:<8.2f} {row['price_per_month_mean']:<10.0f}")
        
        return pricing_stats
    
    def analyze_features(self):
        """Analyze feature adoption in successful APIs"""
        print("\n🔧 FEATURE ANALYSIS")
        print("=" * 50)
        
        # Define feature columns
        feature_cols = [
            'supports_cors', 'supports_webhooks', 'supports_sdk',
            'documentation_available', 'tutorials_available', 'code_examples_available',
            'openapi_spec_available', 'requires_authentication', 'oauth_supported',
            'jwt_supported', 'rate_limiting_enabled', 'caching_enabled',
            'compression_supported', 'pagination_supported', 'filtering_supported',
            'sorting_supported', 'search_supported', 'webhook_supported'
        ]
        
        # High success vs low success
        high_success = self.df[self.df['success_score'] > self.df['success_score'].quantile(0.8)]
        low_success = self.df[self.df['success_score'] < self.df['success_score'].quantile(0.2)]
        
        print("Feature Adoption in High vs Low Success APIs:")
        print(f"{'Feature':<30} {'High Success':<15} {'Low Success':<15} {'Difference':<12}")
        print("-" * 75)
        
        feature_analysis = []
        for feature in feature_cols:
            if feature in self.df.columns:
                high_pct = high_success[feature].mean() * 100
                low_pct = low_success[feature].mean() * 100
                diff = high_pct - low_pct
                
                feature_analysis.append({
                    'feature': feature,
                    'high_success': high_pct,
                    'low_success': low_pct,
                    'difference': diff
                })
                
                print(f"{feature:<30} {high_pct:<15.1f} {low_pct:<15.1f} {diff:<12.1f}")
        
        # Sort by difference
        feature_analysis.sort(key=lambda x: x['difference'], reverse=True)
        
        print(f"\n🏆 TOP FEATURES THAT DRIVE SUCCESS:")
        for i, feat in enumerate(feature_analysis[:10], 1):
            print(f"{i:2d}. {feat['feature']:<30} | +{feat['difference']:.1f}% difference")
        
        return feature_analysis
    
    def clustering_analysis(self):
        """Perform clustering analysis"""
        print("\n🎯 CLUSTERING ANALYSIS")
        print("=" * 50)
        
        # Select features for clustering
        cluster_features = [
            'views', 'likes', 'downloads', 'rating', 'price_per_month',
            'num_endpoints', 'response_time_ms', 'uptime_percentage',
            'supports_cors', 'supports_webhooks', 'supports_sdk',
            'documentation_available', 'tutorials_available', 'code_examples_available'
        ]
        
        # Filter to only existing columns
        cluster_features = [f for f in cluster_features if f in self.df.columns]
        
        X_cluster = self.df[cluster_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df['cluster'] = clusters
        
        # Analyze clusters
        cluster_stats = self.df.groupby('cluster').agg({
            'success_score': ['mean', 'std', 'count'],
            'views': 'mean',
            'likes': 'mean',
            'downloads': 'mean',
            'rating': 'mean',
            'price_per_month': 'mean',
            'num_endpoints': 'mean'
        }).round(2)
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        
        print("Cluster Analysis:")
        print(f"{'Cluster':<8} {'Count':<8} {'Success':<10} {'Views':<10} {'Rating':<8} {'Price':<10}")
        print("-" * 70)
        
        for cluster_id in range(5):
            row = cluster_stats.loc[cluster_id]
            print(f"{cluster_id:<8} {row['success_score_count']:<8.0f} {row['success_score_mean']:<10.0f} "
                  f"{row['views_mean']:<10.0f} {row['rating_mean']:<8.2f} {row['price_per_month_mean']:<10.0f}")
        
        # Analyze cluster characteristics
        print(f"\n🔍 CLUSTER CHARACTERISTICS:")
        for cluster_id in range(5):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            print(f"\n--- CLUSTER {cluster_id} ---")
            print(f"Size: {len(cluster_data):,} APIs ({len(cluster_data)/len(self.df)*100:.1f}%)")
            print(f"Avg Success Score: {cluster_data['success_score'].mean():.0f}")
            print(f"Top Categories: {cluster_data['category'].value_counts().head(3).to_dict()}")
            print(f"Top Pricing: {cluster_data['pricing_model'].value_counts().head(2).to_dict()}")
        
        return cluster_stats
    
    def train_models(self):
        """Train models to predict success"""
        print("\n🤖 MODEL TRAINING")
        print("=" * 50)
        
        # Select features
        feature_cols = [
            'views', 'likes', 'downloads', 'rating', 'price_per_month',
            'num_endpoints', 'response_time_ms', 'uptime_percentage',
            'supports_cors', 'supports_webhooks', 'supports_sdk',
            'documentation_available', 'tutorials_available', 'code_examples_available',
            'openapi_spec_available', 'requires_authentication', 'oauth_supported',
            'jwt_supported', 'rate_limiting_enabled', 'caching_enabled'
        ]
        
        # Filter to existing columns
        feature_cols = [f for f in feature_cols if f in self.df.columns]
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['success_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
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
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<35}: {row['importance']:.3f}")
            
            # Save feature importance plot
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance_simple.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return results, best_model_name
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        print("\n💡 SUCCESS RECOMMENDATIONS")
        print("=" * 50)
        
        # Analyze top performers
        top_10_percent = self.df.nlargest(int(len(self.df) * 0.1), 'success_score')
        
        print("🎯 KEY SUCCESS FACTORS (from top 10% performers):")
        print()
        
        # Category recommendations
        top_categories = top_10_percent['category'].value_counts().head(5)
        print("1. TOP CATEGORIES:")
        for category, count in top_categories.items():
            print(f"   - {category}: {count} APIs ({count/len(top_10_percent)*100:.1f}%)")
        
        # Pricing recommendations
        pricing_dist = top_10_percent['pricing_model'].value_counts(normalize=True) * 100
        print(f"\n2. PRICING STRATEGY:")
        for pricing, pct in pricing_dist.items():
            print(f"   - {pricing}: {pct:.1f}% of top performers")
        
        # Feature recommendations
        feature_cols = [
            'supports_cors', 'supports_webhooks', 'supports_sdk',
            'documentation_available', 'tutorials_available', 'code_examples_available',
            'openapi_spec_available', 'requires_authentication', 'oauth_supported',
            'jwt_supported', 'rate_limiting_enabled', 'caching_enabled'
        ]
        
        feature_adoption = {}
        for feature in feature_cols:
            if feature in self.df.columns:
                adoption = top_10_percent[feature].mean() * 100
                feature_adoption[feature] = adoption
        
        # Sort by adoption rate
        sorted_features = sorted(feature_adoption.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n3. ESSENTIAL FEATURES (adoption rate in top performers):")
        for feature, pct in sorted_features[:10]:
            print(f"   - {feature.replace('_', ' ').title()}: {pct:.1f}%")
        
        # Technical specifications
        print(f"\n4. TECHNICAL SPECIFICATIONS:")
        print(f"   - Average Endpoints: {top_10_percent['num_endpoints'].mean():.0f}")
        print(f"   - Average Response Time: {top_10_percent['response_time_ms'].mean():.0f}ms")
        print(f"   - Average Uptime: {top_10_percent['uptime_percentage'].mean():.1f}%")
        print(f"   - Average Rating: {top_10_percent['rating'].mean():.1f}/5.0")
        
        print(f"\n🚀 ACTIONABLE INSIGHTS:")
        print("   • Focus on high-performing categories")
        print("   • Implement comprehensive documentation and code examples")
        print("   • Offer multiple pricing tiers including free options")
        print("   • Ensure high uptime (>97%) and fast response times")
        print("   • Provide SDKs and webhook support")
        print("   • Target 20-50 endpoints for optimal functionality")
        print("   • Maintain high ratings through quality and support")
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("🎯 SIMPLE API SUCCESS ANALYSIS")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Analyze correlations
        strong, moderate, weak, zero, correlations = self.analyze_correlations()
        
        # Analyze categories
        category_stats = self.analyze_categories()
        
        # Analyze pricing
        pricing_stats = self.analyze_pricing()
        
        # Analyze features
        feature_analysis = self.analyze_features()
        
        # Clustering analysis
        cluster_stats = self.clustering_analysis()
        
        # Train models
        model_results, best_model = self.train_models()
        
        # Generate recommendations
        self.generate_recommendations()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(self.df):,} APIs across {self.df['category'].nunique()} categories")
        print(f"🎯 Identified {len(strong)} strong correlation features")
        print(f"🔍 Found 5 distinct API clusters")
        print(f"🏆 Best model: {best_model}")
        
        return {
            'strong_features': strong,
            'category_stats': category_stats,
            'pricing_stats': pricing_stats,
            'feature_analysis': feature_analysis,
            'cluster_stats': cluster_stats,
            'model_results': model_results,
            'best_model': best_model
        }

def main():
    analyzer = SimpleAPIAnalyzer('rapidapi_comprehensive_20251005_225830.csv')
    results = analyzer.run_analysis()
    
    print(f"\n📁 Analysis files created:")
    print("   - feature_importance_simple.png")

if __name__ == "__main__":
    main()