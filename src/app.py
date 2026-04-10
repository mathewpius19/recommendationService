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
        today = datetime.now()
        user_df = pd.DataFrame([{
        "userId": data["userId"],
        "genrePref": ", ".join(data["genrePref"]),
        "movie_id": i["movieId"],
        "rating": i["rating"],
        "clicks": i["clicks"],
        "watched": i["watched"],
        "days_since_last_interaction": (today - datetime.fromisoformat(i["lastInteraction"])).days,
    } for i in data["interactions"]])
        

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