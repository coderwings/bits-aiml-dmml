"""
Synthetic Data Generator for RecoMart Recommendation System Pipeline.
Generates realistic e-commerce clickstream, transactional purchase history, user profile, and product metadata.
"""

import os
import sys

# Auto-re-execute using ./venv/bin/python if launched with system python3
venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "venv", "bin", "python"))
if os.path.exists(venv_python) and sys.executable != venv_python:
    import subprocess
    sys.exit(subprocess.call([venv_python] + sys.argv))

import random
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(output_dir="input_data", num_users=200, num_items=50, num_interactions=2000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Items Metadata
    categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Beauty & Personal Care"]
    brands = ["Apex", "Nova", "Zenith", "Vortex", "Lumina", "Echo", "Titan"]

    items = []
    item_prices = {}
    for item_id in range(501, 501 + num_items):
        cat = random.choice(categories)
        brand = random.choice(brands)
        price = round(float(np.random.exponential(scale=50) + 10), 2)
        sentiment_score = round(float(np.random.uniform(0.5, 0.99)), 2)
        popularity_score = round(float(np.random.uniform(1.0, 10.0)), 2)
        iid_str = str(item_id)
        item_prices[iid_str] = price
        items.append({
            "item_id": iid_str,
            "product_name": f"{brand} {cat[:-1]} Spec-{item_id}",
            "category": cat,
            "brand": brand,
            "price": price,
            "sentiment_score": sentiment_score,
            "popularity_score": popularity_score
        })

    items_path = os.path.join(output_dir, "items.json")
    with open(items_path, "w") as f:
        json.dump(items, f, indent=4)
    print(f"Generated {len(items)} items at '{items_path}'")

    # 2. Generate User Profiles
    genders = ["M", "F", "Non-Binary"]
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    users = []
    start_date = datetime(2023, 1, 1)

    for user_id in range(1001, 1001 + num_users):
        age = int(np.random.randint(18, 65))
        gender = random.choice(genders)
        signup_days = random.randint(0, 500)
        signup_date = (start_date + timedelta(days=signup_days)).strftime("%Y-%m-%d")
        tier = np.random.choice(tiers, p=[0.5, 0.3, 0.15, 0.05])
        users.append({
            "user_id": str(user_id),
            "age": age,
            "gender": gender,
            "signup_date": signup_date,
            "membership_tier": tier
        })

    users_df = pd.DataFrame(users)
    users_path = os.path.join(output_dir, "users.csv")
    users_df.to_csv(users_path, index=False)
    print(f"Generated {len(users)} users at '{users_path}'")

    # 3. Generate Transactional Purchase History
    interaction_types = ["view", "add_to_cart", "purchase"]
    devices = ["web", "mobile_app"]

    user_ids = [u["user_id"] for u in users]
    item_ids = [i["item_id"] for i in items]

    item_qualities = {i: np.random.uniform(2.5, 5.0) for i in item_ids}

    transactions = []
    base_time = datetime(2024, 1, 1)

    for idx in range(1, num_interactions + 1):
        uid = random.choice(user_ids)
        iid = random.choice(item_ids)
        
        mean_rating = item_qualities[iid]
        rating = round(float(np.clip(np.random.normal(loc=mean_rating, scale=0.8), 1.0, 5.0)), 1)
        
        interaction = np.random.choice(interaction_types, p=[0.5, 0.3, 0.2])
        device = np.random.choice(devices, p=[0.45, 0.55])
        
        unit_price = item_prices[iid]
        quantity = int(np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.25, 0.1, 0.03, 0.02]))
        purchase_price = round(unit_price * random.uniform(0.9, 1.0), 2)
        
        time_offset = timedelta(
            days=random.randint(0, 180),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        purchase_time = (base_time + time_offset).strftime("%Y-%m-%d %H:%M:%S")

        transactions.append({
            "transaction_id": f"TXN_{100000 + idx}",
            "user_id": uid,
            "item_id": iid,
            "purchase_price": purchase_price,
            "quantity": quantity,
            "rating": rating,
            "interaction_type": interaction,
            "device": device,
            "purchase_time": purchase_time,
            "timestamp": purchase_time
        })

    # Add dirty test anomaly rows
    dirty_indices = random.sample(range(len(transactions)), 15)
    for idx in dirty_indices[:5]:
        transactions[idx]["rating"] = 99.0  # Invalid rating out of range
    for idx in dirty_indices[5:10]:
        transactions[idx]["user_id"] = None  # Missing user_id
    for idx in dirty_indices[10:]:
        transactions[idx]["purchase_time"] = "INVALID_DATE"  # Malformed date
        transactions[idx]["timestamp"] = "INVALID_DATE"

    tx_df = pd.DataFrame(transactions)
    tx_path = os.path.join(output_dir, "transactions.csv")
    tx_df.to_csv(tx_path, index=False)
    print(f"Generated {len(tx_df)} transaction records at '{tx_path}'")

if __name__ == "__main__":
    generate_synthetic_data()
