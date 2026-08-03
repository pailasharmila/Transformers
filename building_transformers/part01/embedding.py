import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Converts input token IDs to dense vector embeddings.
    Also multiplies the embeddings by sqrt(d_model) as per the paper.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        print(f"[INIT] d_model: {d_model}, vocab_size: {vocab_size}")

        # Create the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Inspect initial weights
        print("[INIT] Embedding weight shape:", self.embedding.weight.shape)
        print("[INIT] Sample weights (first 2 rows):\n", self.embedding.weight[:2])

    def forward(self, x):
        print("\n[FORWARD] ===== START =====")

        # Input inspection
        print("[FORWARD] Input tensor (token IDs):\n", x)
        print("[FORWARD] Input shape:", x.shape)
        print("[FORWARD] Input dtype:", x.dtype)

        # Step 1: Embedding lookup
        embedded = self.embedding(x)
        print("\n[STEP 1] Raw embeddings (before scaling):")
        print("Shape:", embedded.shape)
        print("Sample values:\n", embedded[0, :2])  # first batch, first 2 tokens

        # Step 2: Scaling factor
        scale = math.sqrt(self.d_model)
        print("\n[STEP 2] Scaling factor sqrt(d_model):", scale)

        # Step 3: Apply scaling
        scaled_embeddings = embedded * scale
        print("\n[STEP 3] Scaled embeddings:")
        print("Shape:", scaled_embeddings.shape)
        print("Sample values:\n", scaled_embeddings[0, :2])

        print("[FORWARD] ===== END =====\n")

        return scaled_embeddings


model = InputEmbeddings(d_model=8, vocab_size=100)

x = torch.tensor([[1, 5, 10], [2, 3, 4]])
output = model(x)