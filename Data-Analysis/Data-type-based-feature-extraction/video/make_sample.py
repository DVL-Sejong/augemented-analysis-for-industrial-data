import json
import random

sample_tweets = [
    "Sunset over Santa Monica Pier hits different after a long week.",
    "I’m always amazed by the disconnect between what we see in the news and the reality of the world around us.",
    "Road-trippin: from Montréal down to Bar Harbor via Route 201. Maine forests already blazing with fall colors.",
    "Chilly morning jog in Central Park. Coffee afterwards from my fav Brooklyn spot.",
    "Visiting Seoul for the first time—so vibrant and fast-paced!",
    "Rainy night in Tokyo, neon reflections everywhere.",
    "Exploring ancient ruins near Cusco. The altitude is no joke!",
    "Quick stop in Prague before heading to Vienna.",
    "Lost in Istanbul’s Grand Bazaar. Worth every minute.",
    "Woke up to views of Mount Fuji from the ryokan."
]

batch_data = []
for i in range(200):
    tweet = random.choice(sample_tweets)
    user_id = f"test_user_{i:03d}"
    batch_data.append({"tweet": tweet, "user_id": user_id})

with open("batch_input.json", "w", encoding="utf-8") as f:
    json.dump(batch_data, f, indent=2, ensure_ascii=False)

print("✅ 200개짜리 batch_input.json 생성 완료")
