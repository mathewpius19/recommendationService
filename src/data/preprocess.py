def buildTextColumn(df):
    if "text" not in df.columns:
        df["text"] = (
            "Title: " + df["title"]+
            ". Genres: "+ df["genres"].fillna("")+
            ". Tags: "+ df["tags"].fillna("") 
        )
    else:
        print("text column already exists. Sample text: ", df["text"].iloc[0])
        return df