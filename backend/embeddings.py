from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text):
    return model.encode(text).tolist()

if __name__ == "__main__":
    text = "Authentication uses PostgreSQL"
    vector = create_embedding(text)
    print(len(vector))
    print(vector[:5])