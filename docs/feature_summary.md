# RecoMart Feature Engineering & Transformation Logic Summary

## 1. User Features (`user_features`)
- `user_interaction_count`: Total number of items clicked/viewed/purchased by user.
- `user_avg_rating`: Mean rating assigned by user across all interactions.
- `user_rating_std`: Standard deviation of user ratings (measures user generosity/strictness).
- `tier_code`: Encoded membership tier (Bronze=1, Silver=2, Gold=3, Platinum=4).
- `last_active_timestamp`: Latest interaction timestamp for user recency calculation.

## 2. Item Features (`item_features`)
- `item_interaction_count`: Popularity frequency measure across all users.
- `item_avg_rating`: Mean implicit/explicit rating score for item.
- `item_rating_std`: Rating variance indicator for quality consistency.
- `price_normalized`: MinMax scaled item price (0.0 to 1.0).
- `sentiment_score`: External API review sentiment score (0.0 to 1.0).
- `popularity_normalized`: MinMax normalized item view/sales popularity score.
- `category_code`: Categorical integer encoding for product category.

## 3. Co-Occurrence Similarity Features (`item_cooccurrence`)
- `cooccurrence_count`: Number of distinct users who interacted positively with both Item A and Item B.
- `jaccard_similarity`: Jaccard index measuring co-interaction ratio \( J(A,B) = \frac{|A \cap B|}{|A \cup B|} \).
