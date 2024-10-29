import matplotlib.pyplot as plt

# Generic training function for both Transformer and LSTM models
def train_model(model, dataloader, optimizer, criterion, num_epochs=100, device='gpu'):
    model.to(device)
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_batch = y_batch.view(-1)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)  # Use external criterion


            # Backward pass and optimization
            loss.backward()
            optimizer.step()


            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_loss:.4f}\n")

    plt.plot(range(num_epochs), loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    print("Training complete.")