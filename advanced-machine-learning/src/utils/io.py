import matplotlib.pyplot as plt

def print_images(dataset, dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(len(images)):
        img = images[i].permute(1, 2, 0).numpy()
        label_idx = labels[i].item()
        label_name = dataset.classes[label_idx]

        axes[i].imshow(img)
        axes[i].set_title(label_name.capitalize(), fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()