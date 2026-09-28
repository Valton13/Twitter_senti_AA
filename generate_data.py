import pandas as pd
import random
from datetime import datetime, timedelta

# Sample metadata
brands = ["AcmeCorp", "TechPulse", "CloudNet", "GlobalPay", "NovaApp"]
categories = ["Product", "Customer Support", "Billing", "App Performance", "Delivery"]

positive_templates = [
    "Absolutely loving the new release from @{brand}! The {category} experience is smooth.",
    "Huge thanks to @{brand} support team for fixing my {category} issue so quickly!",
    "The latest update to @{brand} is fantastic. Great improvement in {category}.",
    "Best decision to switch to @{brand}. Exceptional service in {category}!",
    "Super impressed with @{brand}! Fast response and great usability."
]

negative_templates = [
    "Really disappointed with @{brand}. The {category} is completely broken after the update.",
    "Hey @{brand}, your support is non-existent. Waiting hours for a {category} response!",
    "Worst experience ever with @{brand}. Constant crashes and bugs in {category}.",
    "Avoid @{brand} if you care about reliable {category}. Utter disaster.",
    "System is down again for @{brand}. Unacceptable quality of service."
]

data = []
start_date = datetime.now() - timedelta(days=7)

for i in range(1, 201):  # Generate 200 synthetic tweets
    brand = random.choice(brands)
    category = random.choice(categories)
    is_positive = random.choice([True, False])
    
    template = random.choice(positive_templates if is_positive else negative_templates)
    tweet_text = template.format(brand=brand, category=category.lower())
    
    timestamp = start_date + timedelta(minutes=random.randint(0, 10080))
    
    data.append({
        "tweet_id": 1000 + i,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "brand": brand,
        "category": category,
        "tweet": tweet_text
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("batch_tweets.csv", index=False)
print("`batch_tweets.csv` generated successfully with 200 records!")