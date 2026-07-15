from src.models.cnn import CNN
from src.datasets.dataloader import create_dataloaders


def main() -> None:
    train_loader, _ = create_dataloaders()

    model = CNN(num_classes=2)

    images, labels = next(iter(train_loader))

    outputs = model(images)

    print(f"Input shape : {images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()