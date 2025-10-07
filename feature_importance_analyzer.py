import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, chi2
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_scores = {}
        
    def prepare_data(self):
        """Prepare data for analysis"""
        print("🔧 Preparing data for feature importance analysis...")
        
        # Create multiple success metrics
        self.df['success_score'] = (
            self.df['views'] * 0.3 + 
            self.df['likes'] * 0.25 + 
            self.df['downloads'] * 0.25 + 
            self.df['rating'] * 0.2
        )
        
        # Create individual success metrics
        self.df['views_normalized'] = (self.df['views'] - self.df['views'].min()) / (self.df['views'].max() - self.df['views'].min())
        self.df['likes_normalized'] = (self.df['likes'] - self.df['likes'].min()) / (self.df['likes'].max() - self.df['likes'].min())
        self.df['downloads_normalized'] = (self.df['downloads'] - self.df['downloads'].min()) / (self.df['downloads'].max() - self.df['downloads'].min())
        self.df['rating_normalized'] = (self.df['rating'] - self.df['rating'].min()) / (self.df['rating'].max() - self.df['rating'].min())
        
        # Encode categorical variables
        categorical_cols = ['category', 'pricing_model', 'provider', 'api_type', 
                          'protocol', 'response_format', 'status']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"✅ Data prepared: {len(self.df)} APIs")
        
    def calculate_correlation_importance(self):
        """Calculate correlation-based importance scores"""
        print("\n📊 CORRELATION-BASED IMPORTANCE")
        print("=" * 50)
        
        # Get all numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_features = ['id', 'success_score', 'views', 'likes', 'downloads', 'rating',
                          'views_normalized', 'likes_normalized', 'downloads_normalized', 'rating_normalized']
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        correlation_scores = {}
        
        # Calculate correlations with different success metrics
        success_metrics = ['success_score', 'views_normalized', 'likes_normalized', 
                          'downloads_normalized', 'rating_normalized']
        
        for metric in success_metrics:
            print(f"\nAnalyzing correlations with {metric}:")
            correlations = []
            
            for feature in numeric_features:
                corr = self.df[feature].corr(self.df[metric])
                correlations.append((feature, abs(corr)))  # Use absolute correlation
            
            # Sort by correlation strength
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Normalize scores to 0-100
            if correlations:
                max_corr = correlations[0][1]
                for feature, corr in correlations:
                    normalized_score = (corr / max_corr) * 100 if max_corr > 0 else 0
                    if feature not in correlation_scores:
                        correlation_scores[feature] = []
                    correlation_scores[feature].append(normalized_score)
            
            # Show top 10
            print(f"Top 10 features for {metric}:")
            for i, (feature, score) in enumerate(correlations[:10], 1):
                print(f"{i:2d}. {feature:<35}: {score:.3f}")
        
        # Average correlation scores across all metrics
        avg_correlation_scores = {}
        for feature, scores in correlation_scores.items():
            avg_correlation_scores[feature] = np.mean(scores)
        
        return avg_correlation_scores
    
    def calculate_mutual_information_importance(self):
        """Calculate mutual information-based importance scores"""
        print("\n🔍 MUTUAL INFORMATION IMPORTANCE")
        print("=" * 50)
        
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['id', 'name', 'description', 'category', 'pricing_model', 
                                    'provider', 'api_type', 'protocol', 'response_format', 'status',
                                    'url', 'documentation_url', 'last_updated', 'created_date',
                                    'scraped_at', 'success_score', 'views', 'likes', 'downloads', 'rating',
                                    'views_normalized', 'likes_normalized', 'downloads_normalized', 'rating_normalized',
                                    'endpoints', 'tags', 'features', 'sdk_languages', 'version']]
        
        # Only numeric features
        numeric_features = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = self.df[numeric_features].fillna(0)
        
        mi_scores = {}
        success_metrics = ['success_score', 'views_normalized', 'likes_normalized', 
                          'downloads_normalized', 'rating_normalized']
        
        for metric in success_metrics:
            print(f"\nAnalyzing mutual information with {metric}:")
            y = self.df[metric]
            
            # Calculate mutual information
            mi_values = mutual_info_regression(X, y, random_state=42)
            
            # Create feature-MI mapping
            feature_mi = list(zip(numeric_features, mi_values))
            feature_mi.sort(key=lambda x: x[1], reverse=True)
            
            # Normalize to 0-100
            if feature_mi:
                max_mi = feature_mi[0][1]
                for feature, mi in feature_mi:
                    normalized_score = (mi / max_mi) * 100 if max_mi > 0 else 0
                    if feature not in mi_scores:
                        mi_scores[feature] = []
                    mi_scores[feature].append(normalized_score)
            
            # Show top 10
            print(f"Top 10 features for {metric}:")
            for i, (feature, score) in enumerate(feature_mi[:10], 1):
                print(f"{i:2d}. {feature:<35}: {score:.3f}")
        
        # Average MI scores across all metrics
        avg_mi_scores = {}
        for feature, scores in mi_scores.items():
            avg_mi_scores[feature] = np.mean(scores)
        
        return avg_mi_scores
    
    def calculate_model_importance(self):
        """Calculate importance using multiple ML models"""
        print("\n🤖 MODEL-BASED IMPORTANCE")
        print("=" * 50)
        
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['id', 'name', 'description', 'category', 'pricing_model', 
                                    'provider', 'api_type', 'protocol', 'response_format', 'status',
                                    'url', 'documentation_url', 'last_updated', 'created_date',
                                    'scraped_at', 'success_score', 'views', 'likes', 'downloads', 'rating',
                                    'views_normalized', 'likes_normalized', 'downloads_normalized', 'rating_normalized',
                                    'endpoints', 'tags', 'features', 'sdk_languages', 'version']]
        
        numeric_features = self.df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = self.df[numeric_features].fillna(0)
        
        model_scores = {}
        success_metrics = ['success_score', 'views_normalized', 'likes_normalized', 
                          'downloads_normalized', 'rating_normalized']
        
        for metric in success_metrics:
            print(f"\nTraining models for {metric}:")
            y = self.df[metric]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train multiple models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
            }
            
            for model_name, model in models.items():
                print(f"  Training {model_name}...")
                model.fit(X_train, y_train)
                
                if hasattr(model, 'feature_importances_'):
                    # Get feature importance
                    importance_scores = model.feature_importances_
                    
                    # Normalize to 0-100
                    max_importance = np.max(importance_scores)
                    normalized_scores = (importance_scores / max_importance) * 100 if max_importance > 0 else importance_scores
                    
                    for feature, score in zip(numeric_features, normalized_scores):
                        if feature not in model_scores:
                            model_scores[feature] = []
                        model_scores[feature].append(score)
        
        # Average model scores across all models and metrics
        avg_model_scores = {}
        for feature, scores in model_scores.items():
            avg_model_scores[feature] = np.mean(scores)
        
        return avg_model_scores
    
    def calculate_statistical_importance(self):
        """Calculate statistical significance-based importance"""
        print("\n📈 STATISTICAL SIGNIFICANCE IMPORTANCE")
        print("=" * 50)
        
        # Get numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_features = ['id', 'success_score', 'views', 'likes', 'downloads', 'rating',
                          'views_normalized', 'likes_normalized', 'downloads_normalized', 'rating_normalized']
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        statistical_scores = {}
        success_metrics = ['success_score', 'views_normalized', 'likes_normalized', 
                          'downloads_normalized', 'rating_normalized']
        
        for metric in success_metrics:
            print(f"\nAnalyzing statistical significance with {metric}:")
            y = self.df[metric]
            
            # Calculate statistical measures
            feature_stats = []
            
            for feature in numeric_features:
                x = self.df[feature]
                
                # Calculate various statistical measures
                correlation, p_value = stats.pearsonr(x, y)
                t_stat, t_p_value = stats.ttest_ind(x[y > y.median()], x[y <= y.median()])
                
                # F-statistic for regression
                f_stat, f_p_value = f_regression(x.values.reshape(-1, 1), y.values)
                f_stat = f_stat[0] if len(f_stat) > 0 else 0
                f_p_value = f_p_value[0] if len(f_p_value) > 0 else 1
                
                # Combine measures (lower p-value = higher importance)
                combined_score = abs(correlation) * (1 - p_value) * (1 - t_p_value) * (1 - f_p_value)
                feature_stats.append((feature, combined_score, p_value, t_p_value, f_p_value))
            
            # Sort by combined score
            feature_stats.sort(key=lambda x: x[1], reverse=True)
            
            # Normalize to 0-100
            if feature_stats:
                max_score = feature_stats[0][1]
                for feature, score, p_val, t_p_val, f_p_val in feature_stats:
                    normalized_score = (score / max_score) * 100 if max_score > 0 else 0
                    if feature not in statistical_scores:
                        statistical_scores[feature] = []
                    statistical_scores[feature].append(normalized_score)
            
            # Show top 10
            print(f"Top 10 features for {metric}:")
            for i, (feature, score, p_val, t_p_val, f_p_val) in enumerate(feature_stats[:10], 1):
                print(f"{i:2d}. {feature:<35}: {score:.3f} (p={p_val:.3f})")
        
        # Average statistical scores across all metrics
        avg_statistical_scores = {}
        for feature, scores in statistical_scores.items():
            avg_statistical_scores[feature] = np.mean(scores)
        
        return avg_statistical_scores
    
    def calculate_business_importance(self):
        """Calculate business impact-based importance"""
        print("\n💼 BUSINESS IMPACT IMPORTANCE")
        print("=" * 50)
        
        # Define business-critical features
        business_features = {
            'pricing_model_encoded': 0.25,  # Pricing strategy
            'price_per_month': 0.20,        # Revenue potential
            'price_per_request': 0.15,      # Usage-based revenue
            'free_requests_per_month': 0.10, # Freemium model
            'category_encoded': 0.20,       # Market category
            'provider_encoded': 0.10,       # Brand recognition
            'api_type_encoded': 0.15,       # Technical approach
            'protocol_encoded': 0.10,       # Technical compatibility
            'response_format_encoded': 0.05, # Developer experience
            'status_encoded': 0.10,         # Reliability
            'num_endpoints': 0.15,          # Functionality breadth
            'response_time_ms': 0.20,       # Performance
            'uptime_percentage': 0.25,      # Reliability
            'rate_limit_per_hour': 0.10,    # Scalability
            'rate_limit_per_day': 0.10,     # Usage limits
            'rate_limit_per_month': 0.10,   # Monthly limits
            'requires_authentication': 0.15, # Security
            'supports_cors': 0.10,          # Web compatibility
            'supports_webhooks': 0.15,      # Real-time features
            'supports_sdk': 0.20,           # Developer experience
            'api_key_required': 0.10,       # Security
            'oauth_supported': 0.15,        # Modern auth
            'jwt_supported': 0.15,          # Token-based auth
            'basic_auth_supported': 0.10,   # Simple auth
            'api_key_auth_supported': 0.10, # API key auth
            'bearer_token_supported': 0.10, # Token auth
            'custom_headers_required': 0.05, # Customization
            'ip_whitelist_supported': 0.10, # Security
            'rate_limiting_enabled': 0.15,  # Scalability
            'caching_enabled': 0.15,        # Performance
            'compression_supported': 0.10,  # Performance
            'pagination_supported': 0.10,   # Data handling
            'filtering_supported': 0.10,    # Data filtering
            'sorting_supported': 0.10,      # Data sorting
            'search_supported': 0.15,       # Search functionality
            'webhook_supported': 0.15,      # Real-time
            'sdk_available': 0.20,          # Developer tools
            'documentation_available': 0.25, # Developer experience
            'tutorials_available': 0.20,    # Learning resources
            'code_examples_available': 0.25, # Developer experience
            'postman_collection_available': 0.15, # Testing tools
            'openapi_spec_available': 0.20, # API specification
            'graphql_schema_available': 0.15, # Modern API
            'wadl_available': 0.05,         # Legacy support
            'wsdl_available': 0.05,         # Legacy support
            'raml_available': 0.10,         # API design
            'blueprint_available': 0.10,    # API design
            'mock_server_available': 0.15,  # Development tools
            'testing_environment_available': 0.15, # Development
            'sandbox_environment_available': 0.15, # Development
            'staging_environment_available': 0.10, # Development
            'production_environment_available': 0.20 # Production readiness
        }
        
        # Calculate business importance scores
        business_scores = {}
        for feature, weight in business_features.items():
            if feature in self.df.columns:
                # Base score from business weight
                base_score = weight * 100
                
                # Adjust based on actual adoption in successful APIs
                if self.df[feature].dtype == 'bool' or self.df[feature].nunique() == 2:
                    # Binary feature - check adoption in top performers
                    top_performers = self.df[self.df['success_score'] > self.df['success_score'].quantile(0.8)]
                    adoption_rate = top_performers[feature].mean()
                    business_scores[feature] = base_score * (0.5 + adoption_rate)
                else:
                    # Numeric feature - use as is
                    business_scores[feature] = base_score
        
        return business_scores
    
    def calculate_comprehensive_importance(self):
        """Calculate comprehensive importance scores combining all methods"""
        print("\n🎯 COMPREHENSIVE FEATURE IMPORTANCE CALCULATION")
        print("=" * 60)
        
        # Calculate all importance scores
        correlation_scores = self.calculate_correlation_importance()
        mi_scores = self.calculate_mutual_information_importance()
        model_scores = self.calculate_model_importance()
        statistical_scores = self.calculate_statistical_importance()
        business_scores = self.calculate_business_importance()
        
        # Combine all scores with weights
        weights = {
            'correlation': 0.20,
            'mutual_information': 0.20,
            'model_importance': 0.25,
            'statistical': 0.15,
            'business': 0.20
        }
        
        # Get all unique features
        all_features = set()
        for scores in [correlation_scores, mi_scores, model_scores, statistical_scores, business_scores]:
            all_features.update(scores.keys())
        
        # Calculate weighted average
        comprehensive_scores = {}
        for feature in all_features:
            weighted_score = 0
            total_weight = 0
            
            if feature in correlation_scores:
                weighted_score += correlation_scores[feature] * weights['correlation']
                total_weight += weights['correlation']
            
            if feature in mi_scores:
                weighted_score += mi_scores[feature] * weights['mutual_information']
                total_weight += weights['mutual_information']
            
            if feature in model_scores:
                weighted_score += model_scores[feature] * weights['model_importance']
                total_weight += weights['model_importance']
            
            if feature in statistical_scores:
                weighted_score += statistical_scores[feature] * weights['statistical']
                total_weight += weights['statistical']
            
            if feature in business_scores:
                weighted_score += business_scores[feature] * weights['business']
                total_weight += weights['business']
            
            # Normalize by total weight
            if total_weight > 0:
                comprehensive_scores[feature] = weighted_score / total_weight
            else:
                comprehensive_scores[feature] = 0
        
        # Sort by importance
        sorted_features = sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)
        
        return comprehensive_scores, sorted_features
    
    def create_importance_report(self, comprehensive_scores, sorted_features):
        """Create detailed importance report"""
        print("\n📋 COMPREHENSIVE FEATURE IMPORTANCE REPORT")
        print("=" * 60)
        
        # Create DataFrame for analysis
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance_score': score}
            for feature, score in sorted_features
        ])
        
        # Add categories
        def categorize_feature(feature):
            if 'price' in feature or 'pricing' in feature:
                return 'Pricing & Revenue'
            elif 'category' in feature or 'provider' in feature:
                return 'Market & Brand'
            elif 'endpoint' in feature or 'response' in feature or 'uptime' in feature:
                return 'Performance & Reliability'
            elif 'auth' in feature or 'security' in feature or 'cors' in feature:
                return 'Security & Authentication'
            elif 'documentation' in feature or 'tutorial' in feature or 'example' in feature:
                return 'Developer Experience'
            elif 'webhook' in feature or 'sdk' in feature or 'api' in feature:
                return 'API Features'
            elif 'rate_limit' in feature or 'cache' in feature or 'compression' in feature:
                return 'Scalability & Performance'
            else:
                return 'Other'
        
        importance_df['category'] = importance_df['feature'].apply(categorize_feature)
        
        # Print comprehensive report
        print(f"\n🏆 TOP 30 MOST IMPORTANT FEATURES:")
        print(f"{'Rank':<4} {'Feature':<40} {'Score':<8} {'Category':<25}")
        print("-" * 85)
        
        for i, (_, row) in enumerate(importance_df.head(30).iterrows(), 1):
            print(f"{i:<4} {row['feature']:<40} {row['importance_score']:<8.1f} {row['category']:<25}")
        
        # Category analysis
        print(f"\n📊 IMPORTANCE BY CATEGORY:")
        category_analysis = importance_df.groupby('category').agg({
            'importance_score': ['mean', 'max', 'count']
        }).round(2)
        
        category_analysis.columns = ['avg_score', 'max_score', 'count']
        category_analysis = category_analysis.sort_values('avg_score', ascending=False)
        
        for category, row in category_analysis.iterrows():
            print(f"{category:<25}: Avg={row['avg_score']:6.1f}, Max={row['max_score']:6.1f}, Count={row['count']:3.0f}")
        
        # Save detailed report
        importance_df.to_csv('feature_importance_detailed.csv', index=False)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Top 20 features bar chart
        top_20 = importance_df.head(20)
        plt.subplot(2, 1, 1)
        bars = plt.barh(range(len(top_20)), top_20['importance_score'])
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Most Important Features for API Success')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_20['importance_score'])):
            plt.text(score + 1, i, f'{score:.1f}', va='center', fontsize=8)
        
        # Category distribution
        plt.subplot(2, 1, 2)
        category_scores = importance_df.groupby('category')['importance_score'].mean().sort_values(ascending=True)
        bars = plt.barh(range(len(category_scores)), category_scores.values)
        plt.yticks(range(len(category_scores)), category_scores.index)
        plt.xlabel('Average Importance Score')
        plt.title('Average Importance by Category')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, category_scores.values)):
            plt.text(score + 0.5, i, f'{score:.1f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('comprehensive_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return importance_df
    
    def generate_actionable_insights(self, importance_df):
        """Generate actionable insights based on importance scores"""
        print("\n💡 ACTIONABLE INSIGHTS BASED ON FEATURE IMPORTANCE")
        print("=" * 60)
        
        # High importance features (>70 score)
        high_importance = importance_df[importance_df['importance_score'] > 70]
        print(f"\n🔥 CRITICAL FEATURES (Score > 70): {len(high_importance)} features")
        for _, row in high_importance.iterrows():
            print(f"   • {row['feature']}: {row['importance_score']:.1f} points")
        
        # Medium importance features (40-70 score)
        medium_importance = importance_df[(importance_df['importance_score'] >= 40) & (importance_df['importance_score'] <= 70)]
        print(f"\n⚡ IMPORTANT FEATURES (Score 40-70): {len(medium_importance)} features")
        for _, row in medium_importance.head(10).iterrows():
            print(f"   • {row['feature']}: {row['importance_score']:.1f} points")
        
        # Low importance features (<40 score)
        low_importance = importance_df[importance_df['importance_score'] < 40]
        print(f"\n📉 LOW PRIORITY FEATURES (Score < 40): {len(low_importance)} features")
        print("   Consider deprioritizing these features in your API development")
        
        # Category recommendations
        print(f"\n🎯 CATEGORY-BASED RECOMMENDATIONS:")
        category_priority = importance_df.groupby('category')['importance_score'].mean().sort_values(ascending=False)
        
        for i, (category, avg_score) in enumerate(category_priority.items(), 1):
            priority = "🔥 CRITICAL" if avg_score > 70 else "⚡ IMPORTANT" if avg_score > 40 else "📉 LOW"
            print(f"{i:2d}. {category:<25}: {avg_score:6.1f} points - {priority}")
        
        # ROI recommendations
        print(f"\n💰 ROI-BASED RECOMMENDATIONS:")
        print("   Focus on features with high importance scores for maximum impact")
        print("   Implement features in order of importance score")
        print("   Monitor the impact of each feature on your API's success metrics")
        
        return high_importance, medium_importance, low_importance
    
    def run_comprehensive_analysis(self):
        """Run the complete feature importance analysis"""
        print("🎯 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        print("Calculating numeric importance scores for all features...")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        # Calculate comprehensive importance
        comprehensive_scores, sorted_features = self.calculate_comprehensive_importance()
        
        # Create detailed report
        importance_df = self.create_importance_report(comprehensive_scores, sorted_features)
        
        # Generate actionable insights
        high_imp, medium_imp, low_imp = self.generate_actionable_insights(importance_df)
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {len(comprehensive_scores)} features")
        print(f"🔥 Identified {len(high_imp)} critical features")
        print(f"⚡ Identified {len(medium_imp)} important features")
        print(f"📉 Identified {len(low_imp)} low-priority features")
        
        return {
            'comprehensive_scores': comprehensive_scores,
            'importance_df': importance_df,
            'high_importance': high_imp,
            'medium_importance': medium_imp,
            'low_importance': low_imp,
            'sorted_features': sorted_features
        }

def main():
    analyzer = FeatureImportanceAnalyzer('rapidapi_comprehensive_20251005_225830.csv')
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\n📁 Analysis files created:")
    print("   - feature_importance_detailed.csv")
    print("   - comprehensive_feature_importance.png")
    
    # Show top 10 most important features
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (feature, score) in enumerate(results['sorted_features'][:10], 1):
        print(f"{i:2d}. {feature:<40}: {score:.1f} points")

if __name__ == "__main__":
    main()