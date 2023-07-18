import torch
from model.policy import PolicyModel
from data.loader import DataLoader

# Check if CUDA (GPU) is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the PolicyModel and DataLoader
policy_model = PolicyModel(max_length=1024).to(device)
data_loader = DataLoader(folder_path="./storage/v0.1", tokenizer=policy_model.tokenizer,
                         max_length=policy_model.max_length)

# Get the data in PyTorch DataLoader format
batch_size = 32
train_loader = data_loader.get_data_loader(batch_size=batch_size)

# Define the loss function and optimizer for training
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(policy_model.model.parameters(), lr=1e-4)

# Training loop
epochs = 100
for epoch in range(epochs):
    for batch in train_loader:
        # Move batch data to GPU
        batch = [item.to(device) for item in batch]
        input_ids_state, attention_mask_state, input_ids_action, attention_mask_action, \
            input_ids_option, attention_mask_option, rewards = batch

        # Perform forward pass and calculate the loss
        outputs = policy_model.run_inference(inputs={
            "input_ids": input_ids_state,
            "attention_mask": attention_mask_state,
        })
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.shape[-1]), input_ids_action.view(-1))

        # Perform backpropagation and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring training progress
        print(f"Epoch {epoch+1}/{epochs}, Batch Loss: {loss.item()}")

# Save the trained model if desired
torch.save(policy_model.model.state_dict(), "trained_model.pt")