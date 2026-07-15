import torch
from torch import nn

from src.datasets.dataloader import create_dataloaders
from src.models.cnn import CNN
from src.core.trainer import Trainer


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = create_dataloaders()

    model = CNN(num_classes=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
    )


if __name__ == "__main__":
    main()