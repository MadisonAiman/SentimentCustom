from textblob import TextBlob

# 1. Custom Lexicon with Weights for Specific Phrases
custom_lexicon = {
    "to the moon": (0.9, 0.8),  # Polarity, Subjectivity
    "MOASS": (0.8, 0.9),
    "🚀": (1.0, 1.0),  # Rocket emoji
    "shorts": (-0.5, 0.7),
    "hedgies": (-0.7, 0.6),
    "DRS": (0.6, 0.7),  # Positive sentiment for "Direct Registration System"
    "diamond hands": (0.8, 0.9),
    "shill": (-0.8, 0.7),
    "shills": (-0.8, 0.7),
    "options": (0, 0),
    "option": (0, 0),
    "FTDs": (-0.7, 0.7),
    "FTD": (-0.7, 0.7),
}


# 2. Function to Calculate Custom Sentiment from Lexicon
def calculate_custom_sentiment(text):
    words = text.lower().split()
    total_polarity = 0
    total_subjectivity = 0
    count = 0

    for word in words:
        if word in custom_lexicon:
            polarity, subjectivity = custom_lexicon[word]
            total_polarity += polarity
            total_subjectivity += subjectivity
            count += 1

    # Avoid division by zero
    if count == 0:
        return 0, 0

    # Average sentiment scores
    return total_polarity / count, total_subjectivity / count


# 3. Combined Sentiment Analysis Function
def combined_sentiment_analysis(text):
    # Step 1: Custom lexicon sentiment
    custom_polarity, custom_subjectivity = calculate_custom_sentiment(text)

    # Step 2: TextBlob sentiment
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity

    # Step 3: Combine results (average of custom lexicon and TextBlob)
    final_polarity = (custom_polarity + textblob_polarity) / 2
    final_subjectivity = (custom_subjectivity + textblob_subjectivity) / 2

    return {
        "text": text,
        "custom_sentiment": (custom_polarity, custom_subjectivity),
        "textblob_sentiment": (textblob_polarity, textblob_subjectivity),
        "final_polarity": final_polarity,
        "final_subjectivity": final_subjectivity,
    }


# Example Input
text = """Everyone is talking about RK.

The real significance of the Chicago exchange is Ryan Cohens ability to purchase shares directly from it. In the past when he bought his shares, the Chicago exchange lit up like crazy.

In my humble opinion, it seems as though there is a lockout of some sort for the executives. I hope I’m wrong but I believe Chicago lighting up today is a happy coincidence as I don’t believe Ryan Cohen is able to buy right now." 
""" 
result = combined_sentiment_analysis(text)

# Print Results
print("Original Text:", result["text"])
print("Custom Sentiment:", result["custom_sentiment"])
print("TextBlob Sentiment:", result["textblob_sentiment"])
print("Final Polarity:", result["final_polarity"])
print("Final Subjectivity:", result["final_subjectivity"])
