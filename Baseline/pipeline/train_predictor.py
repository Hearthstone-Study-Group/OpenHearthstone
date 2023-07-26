import torch
from tqdm import tqdm
from model.predictor import PredictorModel
from data.transition import TransitionLoader
# Check if CUDA (GPU) is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the PolicyModel and DataLoader
policy_model = PredictorModel(max_length=900).to(device)
data_loader = TransitionLoader(folder_path="./storage/v0.1", tokenizer=policy_model.tokenizer, difference=True,
                               max_length=policy_model.max_length)

# Get the data in PyTorch DataLoader format
batch_size = 1
train_loader = data_loader.get_data_loader(batch_size=batch_size)

# Define the loss function and optimizer for training
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(policy_model.model.parameters(), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch in (pbar := tqdm(train_loader)):
        # Move batch data to GPU
        batch = [item.to(device) for item in batch]
        input_ids_state, attention_mask_state, input_ids_action, attention_mask_action, \
            input_ids_next_state, attention_mask_next_state, rewards = batch

        # Perform forward pass and calculate the loss
        outputs = policy_model.run_inference(inputs={
            "input_ids": input_ids_state,
            "attention_mask": attention_mask_state,
            "labels": input_ids_next_state
        })
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.shape[-1]), input_ids_next_state.view(-1))

        # Perform backpropagation and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring training progress
        pbar.set_description(f"Loss: {loss.item()}")

# Save the trained model if desired
policy_model.model.save_pretrained("trained/trained_transition_tglobal_0.9k")
policy_model.tokenizer.save_pretrained("trained/trained_transition_tglobal_0.9k")
