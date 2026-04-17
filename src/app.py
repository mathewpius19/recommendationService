from flask import Flask, request, jsonify
from src.service.initialize_services import initialize_services
from src.indexing.search import search_query, search_user_query
from src.util.contentFilter import filter_content_type
import pandas as pd
from datetime import datetime

app = Flask(__name__)

PORT = 4400


@app.route("/recommend/search", methods=["POST"])
def semanticSearchByQuery():
    try:
        data = request.get_json()
        print("data from request ", data)
        query = data["query"].strip()
        print("request received:", query)
        contentTypes = filter_content_type(query)

        results = search_query(contentTypes, [query])

        print("results are ", results)
        return jsonify(results)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/buildUserPreferences", methods = ["POST"])
def buildUserPreferences():
    try:
        data = request.get_json()
        print("data from request ", data)
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        user_id = data.get("userId")
        genre_pref = data.get("genrePref") or []
        interactions = data.get("interactions") or []

        if user_id is None:
            return jsonify({"error": "userId is required"}), 400

        if isinstance(genre_pref, str):
            genre_pref = [genre.strip() for genre in genre_pref.split(",") if genre.strip()]

        today = datetime.now()
        rows = []

        for interaction in interactions:
            last_interaction = interaction.get("lastInteraction")
            if last_interaction:
                days_since_last_interaction = (today - datetime.fromisoformat(last_interaction)).days
            else:
                days_since_last_interaction = 30

            rows.append({
                "userId": user_id,
                "genrePref": ", ".join(genre_pref),
                "movie_id": interaction.get("movieId"),
                "rating": interaction.get("rating", 0),
                "clicks": interaction.get("clicks", 0),
                "watched": interaction.get("watched", False),
                "days_since_last_interaction": days_since_last_interaction,
            })

        if not rows:
            rows.append({
                "userId": user_id,
                "genrePref": ", ".join(genre_pref),
                "movie_id": None,
                "rating": 0,
                "clicks": 0,
                "watched": False,
                "days_since_last_interaction": 30,
            })

        user_df = pd.DataFrame(rows)
        

        print("userDF ", user_df)
        response = search_user_query(user_df)
        result = [int(x) for x in response]
        print("result ", result)
        return jsonify({"movieIds":result})


    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    initialize_services()
    app.run(host="0.0.0.0", port=PORT, debug=True)
