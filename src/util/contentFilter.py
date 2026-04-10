def filter_content_type(query):

    query = query.lower()

    contentTypes = []

    if "documentary" in query or "documentaries" in query:
        contentTypes.append("Documentary")

    if "movie" in query or "movies" in query:
        contentTypes.append("movies")

    return contentTypes