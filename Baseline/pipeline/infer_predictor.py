import torch
from tqdm import tqdm
from model.predictor import PredictorModel
from data.transition import TransitionLoader

# Check if CUDA (GPU) is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the PolicyModel and DataLoader
policy_model = PredictorModel(max_length=1024, pretrained="trained/trained_transition_tglobal_game").to(device)
data_loader = TransitionLoader(folder_path="./storage/v0.1", tokenizer=policy_model.tokenizer, difference=True,
                               max_length=policy_model.max_length)

# Get the data in PyTorch DataLoader format
batch_size = 1
train_loader = data_loader.get_data_loader(batch_size=batch_size)

for batch in (pbar := tqdm(train_loader)):
    # Move batch data to GPU
    batch = [item.to(device) for item in batch]
    input_ids_state, attention_mask_state, input_ids_action, attention_mask_action, \
        input_ids_next_state, attention_mask_next_state, rewards = batch
    print("Input state:")
    input_state_str = policy_model.tokenizer.decode(input_ids_state[0], skip_special_tokens=True)
    print(input_state_str)
    with torch.no_grad():
        # Perform forward pass and calculate the loss
        outputs = policy_model.run_prediction(inputs={
            "input_ids": input_ids_state,
            "attention_mask": attention_mask_state
        })
    print("Predicted state:")
    content = policy_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(content)
    print("Real next state:")
    next_state_str = policy_model.tokenizer.decode(input_ids_next_state[0], skip_special_tokens=True)
    print(next_state_str)
    break

