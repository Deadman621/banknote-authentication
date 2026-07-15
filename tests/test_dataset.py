from src.datasets.dataloader import create_dataloaders

train_loader, _ = create_dataloaders()

for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")

print("✓ Successfully iterated through the entire DataLoader.")