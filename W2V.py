import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset


class CustomDataset(Dataset):
    def __init__(self, file_dir):
        data = torch.load(
            file_dir,
            map_location="cpu",
            weights_only=True
        )

        self.word_to_id = data["word_to_id"]
        self.id_to_word = data["id_to_word"]
        self.training_pairs = data["training_pairs"]
        self.sample_probs = data["sample_probs"]

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, item):
        return self.training_pairs[item]


def get_negative_sample(cd: CustomDataset, num_of_samples, batch_size):
    return torch.multinomial(
        cd.sample_probs,
        num_samples=batch_size * num_of_samples,
        replacement=True
    ).view(batch_size, num_of_samples)


if __name__ == "__main__":

    # 1. Chọn device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 2. Dataset và DataLoader
    cd = CustomDataset("../code/word2vec_data.pt")

    n = min(1_000_000, len(cd))
    indices = torch.randperm(len(cd))[:n].tolist()

    train_dataset = Subset(cd, indices)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    vocab_size = len(cd.word_to_id)
    embedding_dim = 300
    num_of_samples = 4

    # Số epoch muốn train thêm trong lần chạy này
    epochs = 20

    checkpoint_path = "word2vec_checkpoint.pt"

    # 3. Khởi tạo embedding
    Win = nn.Embedding(vocab_size, embedding_dim).to(device)
    Wout = nn.Embedding(vocab_size, embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        list(Win.parameters()) + list(Wout.parameters()),
        lr=0.001
    )

    start_epoch = 0

    # 4. Load checkpoint nếu tồn tại
    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        # Kiểm tra vocabulary
        if checkpoint["word_to_id"] != cd.word_to_id:
            raise ValueError(
                "Vocabulary không khớp với checkpoint. "
                "Không thể tiếp tục training."
            )

        if checkpoint["embedding_dim"] != embedding_dim:
            raise ValueError(
                "Embedding dimension không khớp với checkpoint."
            )

        if checkpoint["vocab_size"] != vocab_size:
            raise ValueError(
                "Vocabulary size không khớp với checkpoint."
            )

        Win.load_state_dict(checkpoint["Win"])
        Wout.load_state_dict(checkpoint["Wout"])

        # Khôi phục trạng thái Adam
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"]

        print("Loaded checkpoint successfully!")
        print("Resuming from epoch:", start_epoch)

    else:
        print("Initialized embeddings randomly")

    # 5. Training
    for epoch in range(start_epoch, start_epoch + epochs):

        Win.train()
        Wout.train()

        total_loss = 0.0

        for batch in train_dataloader:

            batch = batch.to(device)

            centre_ids = batch[:, 0]
            context_ids = batch[:, 1]

            # Forward
            centre_vectors = Win(centre_ids)
            context_vectors = Wout(context_ids)

            right_scores = torch.sum(
                centre_vectors * context_vectors,
                dim=1
            )

            # Negative sampling
            wrong_ids = get_negative_sample(
                cd,
                num_of_samples=num_of_samples,
                batch_size=len(batch)
            ).to(device)

            wrong_vectors = Wout(wrong_ids)

            wrong_scores = torch.sum(
                wrong_vectors * centre_vectors.unsqueeze(1),
                dim=2
            )

            # Loss
            right_loss = -F.logsigmoid(right_scores)

            wrong_loss = -F.logsigmoid(
                -wrong_scores
            ).sum(dim=1)

            loss = (right_loss + wrong_loss).mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        current_epoch = epoch + 1

        print(
            f"Epoch {current_epoch}, "
            f"Loss: {avg_loss:.4f}"
        )

        # 6. Lưu checkpoint sau mỗi epoch
        torch.save({
            "epoch": current_epoch,
            "Win": Win.state_dict(),
            "Wout": Wout.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "word_to_id": cd.word_to_id
        }, checkpoint_path)

    print("Đã lưu checkpoint!")