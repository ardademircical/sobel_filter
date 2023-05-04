import os, sys
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision import transforms


from sobel_filter import SobelFilter


# example: python train_run.py keyword temp_keyword
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Put in training hyperparameters.')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default=None, help='where you store your ImageNet dataset files')
    parser.add_argument('--run_validation', type=bool, default=True, help='enable validation')
    parser.add_argument('--output_dir', type=str, default=None, help='directory where you want to save model weights post training')



    args = parser.parse_args()

    model = SobelFilter()
    model.to("cuda")

    # Preprocessing

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    # Set hyperparameters
    num_epochs =  args.epoch
    lr = args.lr
    batch_size = args.batch_size

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initiate training set
    imagenet_train_dataset = ImageNet(root=args.data_dir, split="train", transform=transform)
    train_dataloader = DataLoader(imagenet_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item()}")

    # save model weights

    output_dir = args.output_dir
    output_weights = output_dir + "/model.pt"
    torch.save(model.state_dict(), output_weights)

    # Check if we want to run validation 
    run_validation = args.run_validation

    if run_validation:
        # Run validation 

        # Initiate validation set

        imagenet_val_dataset = ImageNet(root=args.data_dir, split="validation", transform=transform)
        val_dataloader = DataLoader(imagenet_val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        model.eval()
        running_loss = 0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to("cuda") 
                labels = labels.to("cuda")

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        validation_loss = running_loss / len(val_dataloader)

        print("Validation Loss: {}".format(validation_loss))



