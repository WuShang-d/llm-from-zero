from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, shuffle=True, drop_last=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
