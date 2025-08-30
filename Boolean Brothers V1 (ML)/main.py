# Access from http://localhost:8000/main.html

import os
import googlemaps
from uiaitwo import predict_review_consistency
from dotenv import load_dotenv

# Loading of Google Places API
load_dotenv()
API_KEY = os.getenv("API_KEY")
if(API_KEY is None):
    print("API Key not found.")
    quit()
gmaps = googlemaps.Client(key=API_KEY)

# Function to select location
def selectLocation(placeName):

    # Search for places
    places_result = gmaps.places(query=placeName)

    if len(places_result['results']) == 0:
        return [None, 0]
    else:
        return [places_result['results'],len(places_result['results'])]


# Selected place review
def checkReview(s):
    fields = [
        "name",
        "url",
        "rating",
        "user_ratings_total",
        "reviews",
        "type",
        "editorial_summary"
    ]

    details = gmaps.place(s, fields=fields) 

    result = details.get("result", {})

    name = result.get("name")
    url = result.get("url")
    rating = result.get("rating")
    user_ratings_total = result.get("user_ratings_total")
    
    classification = name + " is a " + result.get("editorial_summary", {}).get("overview", "")
    classi = result.get("types", [])

    classitext = name
    for c in classi:
        if (c != "establishment") and (c != "point_of_interest"):
            classitext += c + " "
    
    if result.get("editorial_summary", {}).get("overview", "") != "":
        classitext = classification
    print(classitext)
    reviews = result.get("reviews", [])

    for item in reviews:
        text = item.get("text", "")
        if len(text) > 200:
            item["trunctext"] = text[:200] + "..."
        else:
            item["trunctext"] = text
    
        item["classification"] = predict_review_consistency(text, classitext)["final_combined_label"].capitalize()
        
        if item["classification"] == "Spam" or item["classification"] == "Advertisement" or item["classification"] == "Irrelevant":
            item["adjusted_rating"] = "Undefined"
        else:
            item["adjusted_rating"] = evaluateReview(item, result, user_ratings_total)

    return [name, rating, reviews, url]

# Evaluate reviews
def evaluateReview(review, allreview, noreview):

    output = round(float(review['rating']) + ((float(allreview['rating']) - float(review['rating']))) * ((float(noreview) - 7) / 7000),1)

    return str(output)