#data analysis
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "/Workspace/Users/enis.caliskan@gsom.polimi.it/Master_insssssh/train.csv"  # Replace with your file path
data = pd.read_csv(url)

# Data Inspection and Cleaning

print(data.dtypes)
print(data.head())
print(data.isnull().sum())

# Fill missing values
data['Product_Category_1'].fillna(0, inplace=True)
data['Product_Category_2'].fillna(0, inplace=True)
data['Product_Category_3'].fillna(0, inplace=True)

# Remove duplicates
print("Number of duplicate rows: ", data.duplicated().sum())
data.drop_duplicates(inplace=True)

# Univariate Analysis
# Categorical Features: Age Distribution
sns.countplot(data=data, x='Age', order=sorted(data['Age'].unique()))
plt.show()

# Numerical Features: Purchase Distribution
sns.kdeplot(data['Purchase'], shade=True, color='g')
plt.show()

# Bivariate and Multivariate Analysis
# Page 17: Age vs. Purchase with Gender as hue, and sorted X-axis
sns.violinplot(data=data, x='Age', y='Purchase', hue='Gender', split=True, order=sorted(data['Age'].unique()))
plt.show()

# Occupation vs. Purchase
sns.barplot(data=data, x='Occupation', y='Purchase')
plt.show()

# Page 21: City Category vs. Purchase using Violin plot instead of Boxen plot
sns.violinplot(data=data, x='City_Category', y='Purchase', palette='Pastel1')
plt.show()

# Feature Engineering and Transformation
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].replace('4+', 4).astype(int)

# Advanced Visualization
# Correlation heatmap (similar to before but adding a diverging palette for better visualization)
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='RdYlBu_r')
plt.show()

# Page 24: Marital Status vs. Purchase with Age as hue
for age_group in sorted(data['Age'].unique()):
    plt.figure(figsize=(10,6))
    subset = data[data['Age'] == age_group]
    sns.pointplot(x='Marital_Status', y='Purchase', hue='Age', data=subset, palette='deep', dodge=True, join=False)
    plt.title(f"Marital Status vs. Purchase for Age Group {age_group}")
    plt.legend(loc='upper right')
    plt.show()

# %% [markdown]
# Cleaning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

print("libraries loaded successfully")

# %%
data_train = pd.read_csv('train.csv')

print("Data loaded successfully")

# %%
#Count including missing data
total = data_train.isnull().sum().sort_values(ascending=False)

#Percent of missing data 
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)

# %%
data_train = data_train.drop('Product_Category_3', axis=1)
print("Product_Category_3 column droped successfully")

# %%
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(pd.DataFrame(data_train['Product_Category_2']))
data_train['Product_Category_2'] = imputer.transform(pd.DataFrame(data_train['Product_Category_2']))
data_train['Product_Category_2'] = np.round(data_train['Product_Category_2'])

print("Product_Category_2 column imputed successfully")

# %%
#Number of null values
print('Number of missing values = ',data_train.isnull().sum().max())

# %%
data_train.dtypes


# %%
import pandas as pd
data_train.to_csv('cleantrain.csv', index=False)







# %% [markdown]
# User-Product Interaction Matrix and Collaborative Filtering

# %%
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv('cleantrain.csv')

# User-product interaction matrix
interaction_matrix = pd.pivot_table(train, values='Purchase', index='User_ID', columns='Product_ID', fill_value=0)  # corrected column name

# Missing values 
interaction_matrix = interaction_matrix.fillna(0)

# User similarity 
user_similarity = cosine_similarity(interaction_matrix)

# DataFrame 
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

# Personalized recommendations for a user
def user_collaborative_filtering(user_id, interaction_matrix, user_similarity_df, n_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    recommendations = []

    for similar_user_id, similarity_score in similar_users.items():  
        user_interactions = interaction_matrix.loc[similar_user_id]
        unrated_products = user_interactions[user_interactions == 0].index
        recommendations.extend(unrated_products)

        if len(recommendations) >= n_recommendations:
            break

    return recommendations[:n_recommendations]

# Example
user_id = interaction_matrix.index[0]  
recommendations = user_collaborative_filtering(user_id, interaction_matrix, user_similarity_df, n_recommendations=5)

print("Recommended Products for User", user_id, ":")
for product_id in recommendations:
    print(product_id)


# %%
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np

train = pd.read_csv('cleantrain.csv')

# User-product interaction matrix
interaction_matrix = pd.pivot_table(train, values='Purchase', index='User_ID', columns='Product_ID', fill_value=0)

# User similarity matrix
user_similarity = cosine_similarity(interaction_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

def predict_rating(user_id, product_id, interaction_matrix, user_similarity_df):
    if product_id not in interaction_matrix.columns:
        return train['Purchase'].mean()  
    
    # Top-N most similar users
    N = 10
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:N+1].index
    mean_rating = interaction_matrix.loc[similar_users, product_id].mean()
    return mean_rating

test_data['predicted_purchase'] = test_data.apply(lambda x: predict_rating(x['User_ID'], x['Product_ID'], interaction_matrix, user_similarity_df), axis=1)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(test_data['Purchase'], test_data['predicted_purchase']))
print(f"Root Mean Squared Error (RMSE): {rmse}")


# %% [markdown]
# Content-Based Product

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleantrain.csv')

# Content-based filtering
product_attributes = train[['Product_Category_1', 'Product_Category_2']].astype(str)
product_attributes['attributes'] = product_attributes.apply(lambda x: ' '.join(x), axis=1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_attributes['attributes'])

def content_based_filtering_efficient(product_id, train, tfidf_matrix, n_recommendations=5):

    product_idx = train[train['Product_ID'] == product_id].index[0]
    product_vector = tfidf_matrix[product_idx]

    sim_scores = cosine_similarity(product_vector, tfidf_matrix).flatten()

    sorted_indices = sim_scores.argsort()[::-1]

    top_indices = sorted_indices[1:n_recommendations + 1]

    top_product_ids = train['Product_ID'].iloc[top_indices].tolist()

    return top_product_ids

# Example
product_id = 'P00233342'
recommendations = content_based_filtering_efficient(product_id, train, tfidf_matrix, n_recommendations=5)

print("Recommended Products for Product ID", product_id, ":")
for recommended_product_id in recommendations:
    print(recommended_product_id)


# %%
import numpy as np
from sklearn.metrics import mean_squared_error

train = pd.read_csv('cleantrain.csv')

def predict_rating_content_based(user_id, product_id, train, tfidf_matrix):
    
    rated_products = train[train['User_ID'] == user_id]['Product_ID'].tolist()
    if not rated_products:
        return train['Purchase'].mean()  

    reference_product_id = rated_products[0]  
    if reference_product_id == product_id:  
        if len(rated_products) > 1:
            reference_product_id = rated_products[1]
        else:
            return train['Purchase'].mean()

    # Similarity between products
    idx_ref = train[train['Product_ID'] == reference_product_id].index[0]
    idx_test = train[train['Product_ID'] == product_id].index[0]
    sim_score = cosine_similarity(tfidf_matrix[idx_ref], tfidf_matrix[idx_test]).flatten()[0]

    # Rating
    actual_rating_for_reference = train[(train['User_ID'] == user_id) & (train['Product_ID'] == reference_product_id)]['Purchase'].values[0]
    return actual_rating_for_reference * sim_score

# Sampling 
train_sampled = train.sample(n=1000, random_state=42)
train_sampled['predicted_purchase'] = train_sampled.apply(
    lambda x: predict_rating_content_based(x['User_ID'], x['Product_ID'], train, tfidf_matrix), axis=1
)

# RMSE 
rmse_sampled = np.sqrt(mean_squared_error(train_sampled['Purchase'], train_sampled['predicted_purchase']))
print(f"Root Mean Squared Error (RMSE) on Sampled Data: {rmse_sampled}")


# %% [markdown]
# Content Based User

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleantrain.csv')

# Columns to string 
user_attributes = train[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']].astype(str).drop_duplicates(subset='User_ID')
user_attributes['attributes'] = user_attributes.apply(lambda x: ' '.join(x[1:]), axis=1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(user_attributes['attributes'])

def user_based_content_filtering(user_id, train, tfidf_matrix, n_recommendations=5):
    user_idx = user_attributes[user_attributes['User_ID'] == user_id].index[0]
    user_vector = tfidf_matrix[user_idx]

    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    sorted_indices = sim_scores.argsort()[::-1]

    top_indices = sorted_indices[1:n_recommendations + 1]

    top_user_ids = user_attributes['User_ID'].iloc[top_indices].tolist()

    recommended_products = []
    for uid in top_user_ids:
        products_bought = train[train['User_ID'] == int(uid)]['Product_ID'].tolist()
        recommended_products.extend(products_bought)

    # Product recommendations
    return list(set(recommended_products))[:n_recommendations]

# Example
user_id = '1000001'  
recommendations = user_based_content_filtering(user_id, train, tfidf_matrix, n_recommendations=5)

print("Recommended Products for User ID", user_id, ":")
for recommended_product_id in recommendations:
    print(recommended_product_id)


# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleantrain.csv')
train.reset_index(drop=True, inplace=True)  

product_attributes = train[['Product_Category_1', 'Product_Category_2']].astype(str)
product_attributes['attributes'] = product_attributes.apply(lambda x: ' '.join(x), axis=1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_attributes['attributes'])



def predict_purchase_content_based(user_id, product_id, train, tfidf_matrix):
   
    users_bought_product = train[train['Product_ID'] == product_id]['User_ID'].tolist()
    users_with_similarity_scores = []

    idx_target_user = train[train['User_ID'] == user_id].index[0]
    target_user_vector = tfidf_matrix[idx_target_user]

    for uid in users_bought_product:
        idx_user = train[train['User_ID'] == uid].index[0]
        user_vector = tfidf_matrix[idx_user]
        sim_score = cosine_similarity(target_user_vector, user_vector).flatten()[0]
        purchase_amount = train[(train['User_ID'] == uid) & (train['Product_ID'] == product_id)]['Purchase'].values[0]
        users_with_similarity_scores.append((sim_score, purchase_amount))

    if not users_with_similarity_scores:
        return train['Purchase'].mean() 

    
    total_similarity = sum([score for score, _ in users_with_similarity_scores])
    total_weighted_purchase = sum([score * amount for score, amount in users_with_similarity_scores])
    
    
    if total_similarity == 0:
        return train['Purchase'].mean()

    return total_weighted_purchase / total_similarity


# Sampling 
train_sampled = train.sample(n=1000, random_state=42)

# Prediction
train_sampled['predicted_purchase'] = train_sampled.apply(
    lambda x: predict_purchase_content_based(x['User_ID'], x['Product_ID'], train, tfidf_matrix), axis=1
)

# RMSE 
rmse_sampled = np.sqrt(mean_squared_error(train_sampled['Purchase'], train_sampled['predicted_purchase']))
print(f"Root Mean Squared Error (RMSE) on Sampled Data: {rmse_sampled}")


# %% [markdown]
# 5045

# %% [markdown]
# Hybrid User and Product Content based

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleantrain.csv')

# Product-based recommendation
product_attributes = train[['Product_Category_1', 'Product_Category_2']].astype(str)
product_attributes['attributes'] = product_attributes.apply(lambda x: ' '.join(x), axis=1)
product_tfidf_vectorizer = TfidfVectorizer()
product_tfidf_matrix = product_tfidf_vectorizer.fit_transform(product_attributes['attributes'])

# User-based recommendation
user_attributes = train[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']].astype(str).drop_duplicates(subset='User_ID')
user_attributes['attributes'] = user_attributes.apply(lambda x: ' '.join(x[1:]), axis=1)
user_tfidf_vectorizer = TfidfVectorizer()
user_tfidf_matrix = user_tfidf_vectorizer.fit_transform(user_attributes['attributes'])

def content_based_filtering_efficient(product_id, train, product_tfidf_matrix, n_recommendations=5):
    idx = train.index[train['Product_ID'] == product_id][0]
    cosine_similarities = cosine_similarity(product_tfidf_matrix[idx:idx+1], product_tfidf_matrix).flatten()
    related_products_indices = cosine_similarities.argsort()[-n_recommendations-1:-1][::-1]
    return [train['Product_ID'].iloc[i] for i in related_products_indices]

def user_based_content_filtering(user_id, train, user_tfidf_matrix, n_recommendations=5):
    idx = user_attributes.index[user_attributes['User_ID'] == user_id].tolist()[0]
    cosine_similarities = cosine_similarity(user_tfidf_matrix[idx:idx+1], user_tfidf_matrix).flatten()
    similar_users = cosine_similarities.argsort()[-n_recommendations-1:-1][::-1]
    similar_users_ids = [user_attributes['User_ID'].iloc[i] for i in similar_users]
    recommended_products = []
    for similar_user_id in similar_users_ids:
        user_purchased_products = train[train['User_ID'] == int(similar_user_id)]['Product_ID'].tolist()
        recommended_products.extend(user_purchased_products)
    
    return list(set(recommended_products))



def hybrid_recommendation(user_id, train, user_tfidf_matrix, product_tfidf_matrix, n_recommendations=5):
    
    user_purchased_products = train[train['User_ID'] == int(user_id)]['Product_ID'].tolist()

    product_recommendations = []
    for product in user_purchased_products:
        similar_products = content_based_filtering_efficient(product, train, product_tfidf_matrix, n_recommendations)
        product_recommendations.extend(similar_products)

    user_recommendations = user_based_content_filtering(user_id, train, user_tfidf_matrix, n_recommendations)

    combined_recommendations = list(set(product_recommendations + user_recommendations))

    return combined_recommendations[:n_recommendations]

# Example
user_id = '1000001'
recommendations = hybrid_recommendation(user_id, train, user_tfidf_matrix, product_tfidf_matrix, n_recommendations=5)
print("Recommended Products for User ID", user_id, ":")
for recommended_product_id in recommendations:
    print(recommended_product_id)


# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


train = pd.read_csv('cleantrain.csv')

average_purchase = train.groupby('Product_ID')['Purchase'].mean()

global_average_purchase = train['Purchase'].mean()

def predict_purchase_hybrid(user_id, product_id):
    recommended_products = hybrid_recommendation(user_id, train, user_tfidf_matrix, product_tfidf_matrix)
    if product_id in recommended_products:

        return average_purchase.get(product_id, global_average_purchase)
    else:
        
        return global_average_purchase

# Sample 
train_sampled = train.sample(n=1000, random_state=42)

# Prediction
train_sampled['predicted_purchase'] = train_sampled.apply(
    lambda x: predict_purchase_hybrid(str(x['User_ID']), x['Product_ID']), axis=1
)

# RMSE 
rmse_sampled = np.sqrt(mean_squared_error(train_sampled['Purchase'], train_sampled['predicted_purchase']))
print(f"Root Mean Squared Error (RMSE) on Sampled Data: {rmse_sampled}")

# %% [markdown]
# Hybrid Collaborative and Content

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleantrain.csv')

# Content-Based Filtering 
product_attributes = train[['Product_Category_1', 'Product_Category_2']].astype(str)
product_attributes['attributes'] = product_attributes.apply(lambda x: ' '.join(x), axis=1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_attributes['attributes'])

def content_based_filtering(product_id, tfidf_matrix):
    product_idx = train[train['Product_ID'] == product_id].index[0]
    product_vector = tfidf_matrix[product_idx]
    sim_scores = cosine_similarity(product_vector, tfidf_matrix).flatten()
    return sim_scores

# Collaborative Filtering 
interaction_matrix = pd.pivot_table(train, values='Purchase', index='User_ID', columns='Product_ID', fill_value=0)
user_similarity = cosine_similarity(interaction_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

def collaborative_filtering(user_id):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:].index
    recommended_products = []
    for similar_user in similar_users:
        similar_user_products = interaction_matrix.loc[similar_user]
        for product_id, purchase in similar_user_products.items():
            if purchase > 0:  
                recommended_products.append(product_id)
    return recommended_products[:5]  

def predict_rating_hybrid(user_id, product_id):
    weight_cf = 0.5  
    weight_cbf = 0.5  
    return (weight_cf * predict_rating_collaborative(user_id, product_id)) + (weight_cbf * predict_rating_content_based(user_id, product_id))

def hybrid_recommendation(user_id, tfidf_matrix):
    
    all_products = interaction_matrix.columns.tolist()

    product_ratings = {}
    for product in all_products:
        product_ratings[product] = predict_rating_hybrid(user_id, product)

    sorted_products = sorted(product_ratings, key=product_ratings.get, reverse=True)
    
    return sorted_products[:5]
def predict_rating_collaborative(user_id, product_id):
    
    import random
    return random.random()

def predict_rating_content_based(user_id, product_id):
    import random
    return random.random()


user_id = interaction_matrix.index[0]
recommendations = hybrid_recommendation(user_id, tfidf_matrix)

print("Hybrid Recommended Products for User", user_id, ":")
for product_id in recommendations:
    print(product_id)


# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


train = pd.read_csv('cleantrain.csv')

# Content-Based Filtering setup
product_attributes = train[['Product_Category_1', 'Product_Category_2']].astype(str)
product_attributes['attributes'] = product_attributes.apply(lambda x: ' '.join(x), axis=1)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(product_attributes['attributes'])

# Collaborative Filtering setup
interaction_matrix = pd.pivot_table(train, values='Purchase', index='User_ID', columns='Product_ID', fill_value=0)
user_similarity = cosine_similarity(interaction_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

average_purchase = train.groupby('Product_ID')['Purchase'].mean()


global_average_purchase = train['Purchase'].mean()

def predict_purchase_hybrid(user_id, product_id):
    recommended_products = hybrid_recommendation(user_id, tfidf_matrix)
    if product_id in recommended_products:
        return average_purchase.get(product_id, global_average_purchase)
    else:
        return global_average_purchase

# Sample 
train_sampled = train.sample(n=1000, random_state=42)

# Prediction
train_sampled['predicted_purchase'] = train_sampled.apply(
    lambda x: predict_purchase_hybrid(x['User_ID'], x['Product_ID']), axis=1
)

# RMSE 
rmse_sampled = np.sqrt(mean_squared_error(train_sampled['Purchase'], train_sampled['predicted_purchase']))
print(f"Root Mean Squared Error (RMSE) on Sampled Data: {rmse_sampled}")


# %% [markdown]
# Implementation

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


train = pd.read_csv('cleantrain.csv')


product_attributes = train[['Product_Category_1', 'Product_Category_2']].astype(str)
product_attributes['attributes'] = product_attributes.apply(lambda x: ' '.join(x), axis=1)
product_tfidf_vectorizer = TfidfVectorizer()
product_tfidf_matrix = product_tfidf_vectorizer.fit_transform(product_attributes['attributes'])


user_attributes = train[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']].astype(str).drop_duplicates(subset='User_ID')
user_attributes['attributes'] = user_attributes.apply(lambda x: ' '.join(x[1:]), axis=1)
user_tfidf_vectorizer = TfidfVectorizer()
user_tfidf_matrix = user_tfidf_vectorizer.fit_transform(user_attributes['attributes'])

def content_based_filtering_efficient(product_id, train, product_tfidf_matrix, n_recommendations=5):
    idx = train.index[train['Product_ID'] == product_id][0]
    cosine_similarities = cosine_similarity(product_tfidf_matrix[idx:idx+1], product_tfidf_matrix).flatten()
    related_products_indices = cosine_similarities.argsort()[-n_recommendations-1:-1][::-1]
    return [train['Product_ID'].iloc[i] for i in related_products_indices]

def user_based_content_filtering(user_id, train, user_tfidf_matrix, n_recommendations=5):
    idx = user_attributes.index[user_attributes['User_ID'] == user_id].tolist()[0]
    cosine_similarities = cosine_similarity(user_tfidf_matrix[idx:idx+1], user_tfidf_matrix).flatten()
    similar_users = cosine_similarities.argsort()[-n_recommendations-1:-1][::-1]
    similar_users_ids = [user_attributes['User_ID'].iloc[i] for i in similar_users]
    recommended_products = []
    for similar_user_id in similar_users_ids:
        user_purchased_products = train[train['User_ID'] == int(similar_user_id)]['Product_ID'].tolist()
        recommended_products.extend(user_purchased_products)
    
    return list(set(recommended_products))

def user_based_content_filtering(user_id, train, user_tfidf_matrix, n_recommendations=5):
    user_indices = user_attributes.index[user_attributes['User_ID'] == user_id].tolist()
    
    if not user_indices:  
        return []

    idx = user_indices[0]
    cosine_similarities = cosine_similarity(user_tfidf_matrix[idx:idx+1], user_tfidf_matrix).flatten()
    similar_users = cosine_similarities.argsort()[-n_recommendations-1:-1][::-1]
    similar_users_ids = [user_attributes['User_ID'].iloc[i] for i in similar_users]
    
    recommended_products = []
    for similar_user_id in similar_users_ids:
        user_purchased_products = train[train['User_ID'] == int(similar_user_id)]['Product_ID'].tolist()
        recommended_products.extend(user_purchased_products)
    
    return list(set(recommended_products))

def hybrid_recommendation(user_id, train, user_tfidf_matrix, product_tfidf_matrix, n_recommendations=5):
    
    user_purchased_products = train[train['User_ID'] == int(user_id)]['Product_ID'].tolist()

    product_recommendations = []
    for product in user_purchased_products:
        similar_products = content_based_filtering_efficient(product, train, product_tfidf_matrix, n_recommendations)
        product_recommendations.extend(similar_products)

    user_recommendations = user_based_content_filtering(user_id, train, user_tfidf_matrix, n_recommendations)

    combined_recommendations = list(set(product_recommendations + user_recommendations))

    return combined_recommendations[:n_recommendations]


test = pd.read_csv('testcut.csv')


test['Recommended_Product_1'] = ''
test['Recommended_Product_2'] = ''
test['Recommended_Product_3'] = ''
test['Recommended_Product_4'] = ''
test['Recommended_Product_5'] = ''


for idx, row in test.iterrows():
    user_id = row['User_ID']
    recommendations = hybrid_recommendation(user_id, train, user_tfidf_matrix, product_tfidf_matrix, n_recommendations=5)

    while len(recommendations) < 5:
        recommendations.append('N/A')

    test.at[idx, 'Recommended_Product_1'] = recommendations[0]
    test.at[idx, 'Recommended_Product_2'] = recommendations[1]
    test.at[idx, 'Recommended_Product_3'] = recommendations[2]
    test.at[idx, 'Recommended_Product_4'] = recommendations[3]
    test.at[idx, 'Recommended_Product_5'] = recommendations[4]


test.to_csv('test_recommendations.csv', index=False)



