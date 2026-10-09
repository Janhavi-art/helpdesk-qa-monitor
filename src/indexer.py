from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """
    Convert chunk text into numerical embeddings.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    print("Embedding model loaded successfully.")

    test_text = [
        "My Wi-Fi keeps disconnecting.",
        "I forgot my password."
    ]

    embeddings = model.encode(test_text)

    print("Number of embeddings:", len(embeddings))
    print("Embedding dimensions:", len(embeddings[0]))