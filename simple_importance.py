import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_feature_importance():
    """Calculate numeric importance scores for API features"""
    print("🎯 FEATURE IMPORTANCE CALCULATOR")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('rapidapi_comprehensive_20251005_225830.csv')
    print(f"📊 Loaded {len(df):,} APIs")
    
    # Create success score
    df['success_score'] = (
        df['views'] * 0.3 + 
        df['likes'] * 0.25 + 
        df['downloads'] * 0.25 + 
        df['rating'] * 0.2
    )
    
    # Encode categorical variables
    categorical_cols = ['category', 'pricing_model', 'provider', 'api_type', 
                      'protocol', 'response_format', 'status']
    
    for col in categorical_cols:
        df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    
    # Select numeric features for analysis
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
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df['success_score']
    
    print(f"🔍 Analyzing {len(feature_cols)} features...")
    
    # Method 1: Correlation Analysis
    print("\n📊 CORRELATION ANALYSIS")
    print("-" * 30)
    correlations = {}
    for feature in feature_cols:
        corr = df[feature].corr(df['success_score'])
        correlations[feature] = abs(corr)  # Use absolute correlation
    
    # Method 2: Random Forest Feature Importance
    print("\n🌲 RANDOM FOREST ANALYSIS")
    print("-" * 30)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = dict(zip(feature_cols, rf.feature_importances_))
    
    # Method 3: Mutual Information
    print("\n🔍 MUTUAL INFORMATION ANALYSIS")
    print("-" * 30)
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_importance = dict(zip(feature_cols, mi_scores))
    
    # Method 4: Statistical Significance
    print("\n📈 STATISTICAL SIGNIFICANCE")
    print("-" * 30)
    statistical_scores = {}
    for feature in feature_cols:
        # T-test between high and low success groups
        high_success = df[df['success_score'] > df['success_score'].quantile(0.8)][feature]
        low_success = df[df['success_score'] < df['success_score'].quantile(0.2)][feature]
        
        if len(high_success) > 0 and len(low_success) > 0:
            t_stat, p_value = stats.ttest_ind(high_success, low_success)
            # Convert p-value to importance (lower p-value = higher importance)
            statistical_scores[feature] = 1 - p_value if not np.isnan(p_value) else 0
        else:
            statistical_scores[feature] = 0
    
    # Method 5: Business Impact (manual weights)
    print("\n💼 BUSINESS IMPACT ANALYSIS")
    print("-" * 30)
    business_weights = {
        'pricing_model_encoded': 0.25,
        'price_per_month': 0.20,
        'category_encoded': 0.20,
        'documentation_available': 0.25,
        'tutorials_available': 0.20,
        'code_examples_available': 0.25,
        'openapi_spec_available': 0.20,
        'supports_sdk': 0.20,
        'supports_webhooks': 0.15,
        'requires_authentication': 0.15,
        'oauth_supported': 0.15,
        'jwt_supported': 0.15,
        'rate_limiting_enabled': 0.15,
        'caching_enabled': 0.15,
        'search_supported': 0.15,
        'webhook_supported': 0.15,
        'sdk_available': 0.20,
        'uptime_percentage': 0.25,
        'response_time_ms': 0.20,
        'num_endpoints': 0.15,
        'rating': 0.30
    }
    
    # Normalize all scores to 0-100 scale
    def normalize_scores(scores):
        if not scores:
            return {}
        max_score = max(scores.values())
        return {k: (v / max_score) * 100 if max_score > 0 else 0 for k, v in scores.items()}
    
    # Normalize each method
    corr_norm = normalize_scores(correlations)
    rf_norm = normalize_scores(rf_importance)
    mi_norm = normalize_scores(mi_importance)
    stat_norm = normalize_scores(statistical_scores)
    business_norm = normalize_scores(business_weights)
    
    # Calculate weighted average (equal weights for now)
    weights = {
        'correlation': 0.20,
        'random_forest': 0.25,
        'mutual_info': 0.20,
        'statistical': 0.15,
        'business': 0.20
    }
    
    # Calculate comprehensive scores
    comprehensive_scores = {}
    for feature in feature_cols:
        score = 0
        total_weight = 0
        
        if feature in corr_norm:
            score += corr_norm[feature] * weights['correlation']
            total_weight += weights['correlation']
        
        if feature in rf_norm:
            score += rf_norm[feature] * weights['random_forest']
            total_weight += weights['random_forest']
        
        if feature in mi_norm:
            score += mi_norm[feature] * weights['mutual_info']
            total_weight += weights['mutual_info']
        
        if feature in stat_norm:
            score += stat_norm[feature] * weights['statistical']
            total_weight += weights['statistical']
        
        if feature in business_norm:
            score += business_norm[feature] * weights['business']
            total_weight += weights['business']
        
        if total_weight > 0:
            comprehensive_scores[feature] = score / total_weight
        else:
            comprehensive_scores[feature] = 0
    
    # Sort by importance
    sorted_features = sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create results DataFrame
    results = []
    for feature, score in sorted_features:
        results.append({
            'feature': feature,
            'importance_score': round(score, 1),
            'correlation': round(corr_norm.get(feature, 0), 1),
            'random_forest': round(rf_norm.get(feature, 0), 1),
            'mutual_info': round(mi_norm.get(feature, 0), 1),
            'statistical': round(stat_norm.get(feature, 0), 1),
            'business': round(business_norm.get(feature, 0), 1)
        })
    
    results_df = pd.DataFrame(results)
    
    # Print comprehensive report
    print("\n🎯 COMPREHENSIVE FEATURE IMPORTANCE SCORES")
    print("=" * 80)
    print(f"{'Rank':<4} {'Feature':<35} {'Total':<6} {'Corr':<6} {'RF':<6} {'MI':<6} {'Stat':<6} {'Bus':<6}")
    print("-" * 80)
    
    for i, row in results_df.iterrows():
        print(f"{i+1:<4} {row['feature']:<35} {row['importance_score']:<6.1f} "
              f"{row['correlation']:<6.1f} {row['random_forest']:<6.1f} "
              f"{row['mutual_info']:<6.1f} {row['statistical']:<6.1f} {row['business']:<6.1f}")
    
    # Categorize features
    def categorize_feature(feature):
        if 'price' in feature or 'pricing' in feature:
            return 'Pricing & Revenue'
        elif 'documentation' in feature or 'tutorial' in feature or 'example' in feature:
            return 'Developer Experience'
        elif 'auth' in feature or 'security' in feature:
            return 'Security & Auth'
        elif 'endpoint' in feature or 'response' in feature or 'uptime' in feature:
            return 'Performance'
        elif 'webhook' in feature or 'sdk' in feature:
            return 'API Features'
        elif 'category' in feature or 'provider' in feature:
            return 'Market & Brand'
        else:
            return 'Other'
    
    results_df['category'] = results_df['feature'].apply(categorize_feature)
    
    # Category analysis
    print(f"\n📊 IMPORTANCE BY CATEGORY:")
    category_analysis = results_df.groupby('category').agg({
        'importance_score': ['mean', 'max', 'count']
    }).round(1)
    
    category_analysis.columns = ['avg_score', 'max_score', 'count']
    category_analysis = category_analysis.sort_values('avg_score', ascending=False)
    
    for category, row in category_analysis.iterrows():
        print(f"{category:<20}: Avg={row['avg_score']:6.1f}, Max={row['max_score']:6.1f}, Count={row['count']:3.0f}")
    
    # Save results
    results_df.to_csv('feature_importance_scores.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Top 20 features
    top_20 = results_df.head(20)
    plt.subplot(2, 1, 1)
    bars = plt.barh(range(len(top_20)), top_20['importance_score'])
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('Importance Score (0-100)')
    plt.title('Top 20 Most Important Features for API Success')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_20['importance_score'])):
        plt.text(score + 1, i, f'{score:.1f}', va='center', fontsize=8)
    
    # Category distribution
    plt.subplot(2, 1, 2)
    category_scores = results_df.groupby('category')['importance_score'].mean().sort_values(ascending=True)
    bars = plt.barh(range(len(category_scores)), category_scores.values)
    plt.yticks(range(len(category_scores)), category_scores.index)
    plt.xlabel('Average Importance Score')
    plt.title('Average Importance by Category')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, category_scores.values)):
        plt.text(score + 1, i, f'{score:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('feature_importance_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"🔥 CRITICAL FEATURES (Score > 70): {len(results_df[results_df['importance_score'] > 70])} features")
    print(f"⚡ IMPORTANT FEATURES (Score 40-70): {len(results_df[(results_df['importance_score'] >= 40) & (results_df['importance_score'] <= 70)])} features")
    print(f"📉 LOW PRIORITY (Score < 40): {len(results_df[results_df['importance_score'] < 40])} features")
    
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for i, row in results_df.head(10).iterrows():
        print(f"{i+1:2d}. {row['feature']:<35}: {row['importance_score']:6.1f} points")
    
    print(f"\n✅ Analysis complete! Files saved:")
    print("   - feature_importance_scores.csv")
    print("   - feature_importance_visualization.png")
    
    return results_df

if __name__ == "__main__":
    results = calculate_feature_importance()