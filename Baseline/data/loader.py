import os
import json
import torch
from transformers import GPT2Tokenizer
from definition.GAME_TAG import GAME_TAG


class DataLoader:
    scope = [
        GAME_TAG.ENTITY_ID,
        GAME_TAG.HEALTH,
        GAME_TAG.ATK,
        GAME_TAG.COST,
        GAME_TAG.ARMOR,
        GAME_TAG.PREMIUM,
        GAME_TAG.PLAYSTATE,
        GAME_TAG.LAST_AFFECTED_BY,
        GAME_TAG.STEP,
        GAME_TAG.TURN,
        GAME_TAG.FATIGUE,
        GAME_TAG.FIRST_PLAYER,
        GAME_TAG.RESOURCES_USED,
        GAME_TAG.RESOURCES,
        GAME_TAG.HERO_ENTITY,
        GAME_TAG.MAXHANDSIZE,
        GAME_TAG.PLAYER_ID,
        GAME_TAG.TEAM_ID,
        GAME_TAG.ATTACKING,
        GAME_TAG.ATTACHED,
        GAME_TAG.EXHAUSTED,
        GAME_TAG.DAMAGE,
        GAME_TAG.ZONE,
        GAME_TAG.CONTROLLER,
        GAME_TAG.OWNER,
        GAME_TAG.CARD_SET,
        GAME_TAG.CARDTEXT,
        GAME_TAG.DURABILITY,
        GAME_TAG.SILENCED,
        GAME_TAG.WINDFURY,
        GAME_TAG.TAUNT,
        GAME_TAG.STEALTH,
        GAME_TAG.SPELLPOWER,
        GAME_TAG.DIVINE_SHIELD,
        GAME_TAG.INSPIRE,
        GAME_TAG.POISONOUS,
        GAME_TAG.LIFESTEAL,
        GAME_TAG.RUSH,
        GAME_TAG.OVERKILL,
        GAME_TAG.PROPHECY,
        GAME_TAG.REBORN,
        GAME_TAG.TWINSPELL,
        GAME_TAG.AVENGE,
        GAME_TAG.CHARGE,
        GAME_TAG.NEXT_STEP,
        GAME_TAG.CLASS,
        GAME_TAG.CARDRACE,
        GAME_TAG.CARDTYPE,
        GAME_TAG.RARITY,
        GAME_TAG.STATE,
        GAME_TAG.SUMMONED,
        GAME_TAG.FREEZE,
        GAME_TAG.ENRAGED,
        GAME_TAG.OVERLOAD,
        GAME_TAG.LOYALTY,
        GAME_TAG.DEATHRATTLE,
        GAME_TAG.BATTLECRY,
        GAME_TAG.SECRET,
        GAME_TAG.MAGNET,
        GAME_TAG.COMBO,
        GAME_TAG.IMMUNE,
        GAME_TAG.FROZEN,
        GAME_TAG.ZONE_POSITION,
        GAME_TAG.CARD_TARGET,
        GAME_TAG.NUM_TURNS_IN_PLAY,
        GAME_TAG.NUM_TURNS_LEFT,
        GAME_TAG.NUM_TURNS_IN_HAND,
        GAME_TAG.CURRENT_SPELLPOWER,
        GAME_TAG.TEMP_RESOURCES,
        GAME_TAG.OVERLOAD_OWED,
        GAME_TAG.NUM_ATTACKS_THIS_TURN
    ]

    def __init__(self, folder_path, tokenizer):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.max_length = 1024

    def load_json_data(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def collate_sequence_data(self, data):
        # Collate state, action, and option data and add the reward term
        sequence_data = []
        result = data["metadata"]["result"]
        decay_factor = 0.9  # Decaying factor for the reward
        for state, action, option in zip(data["sequence"]["state"], data["sequence"]["action"], data["sequence"]["option"]):
            reward = result * decay_factor
            collated_data = (state, action, option, reward)
            sequence_data.append(collated_data)
            decay_factor *= decay_factor  # Applying the decay factor recursively
        return sequence_data

    def preprocess_state(self, state):
        state = [entity for entity in state if not (len(entity["card_name"]) == 0 or
                                                    len(entity["card_description"]) == 0 or
                                                    len(entity["card_id"]) == 0 or
                                                    entity["card_id"] is None)]
        for entity in state:
            for tag in list(entity["tags"]):
                if tag not in self.scope:
                    entity["tags"].pop(tag, None)
        return state

    def preprocess_input(self, item):
        ret = json.dumps(item, separators=(',', ':')).replace("\"", "")
        # if len(ret) > 4000:
        #     print(json.dumps(item, separators=(',', ':')))
        keywords = [
            "entity",
            "type",
            "sub_options",
            "sub_option",
            "position",
            "targets",
            "target",
            "card_id",
            "card_name",
            "card_description",
            "tags"
        ]
        for keyword in keywords:
            ret = ret.replace(keyword + ":", "")

        return ret

    def check_data(self, sequence_data):
        # Check the collated data for model input
        data = []
        for state, action, option, reward in sequence_data:
            state = self.preprocess_state(state)
            tokenized_state = self.tokenizer(self.preprocess_input(state),
                                             max_length=self.max_length)
            tokenized_action = self.tokenizer(self.preprocess_input(action),
                                              max_length=self.max_length)
            tokenized_option = self.tokenizer(self.preprocess_input(option),
                                              max_length=self.max_length)
            item = [len(tokenized_state["input_ids"]),
                    len(tokenized_action["input_ids"]),
                    len(tokenized_option["input_ids"])]
            data.append(item)
        return data

    def tokenize_data(self, sequence_data):
        # Tokenize the collated data for model input
        tokenized_data = []
        for state, action, option, reward in sequence_data:
            # print(json.dumps(state, separators=(',', ':')))
            # Perform tokenization for state, action, and option (you may need to adjust this based on your data structure)
            state = self.preprocess_state(state)
            tokenized_state = self.tokenizer(self.preprocess_input(state), return_tensors="pt",
                                             max_length=self.max_length, padding='max_length')
            tokenized_action = self.tokenizer(self.preprocess_input(action), return_tensors="pt",
                                              max_length=self.max_length, padding='max_length')
            tokenized_option = self.tokenizer(self.preprocess_input(option), return_tensors="pt",
                                              max_length=self.max_length, padding='max_length')

            # Combine the tokenized data and add the reward term
            combined_data = {
                "input_ids_state": tokenized_state.input_ids,
                "attention_mask_state": tokenized_state.attention_mask,
                "input_ids_action": tokenized_action.input_ids,
                "attention_mask_action": tokenized_action.attention_mask,
                "input_ids_option": tokenized_option.input_ids,
                "attention_mask_option": tokenized_option.attention_mask,
                "reward": reward
            }
            tokenized_data.append(combined_data)
        return tokenized_data

    def get_data_loader(self, batch_size=32):
        # Load JSON files, collate data, and tokenize for model input
        data = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            json_data = self.load_json_data(file_path)
            sequence_data = self.collate_sequence_data(json_data)
            tokenized_data = self.tokenize_data(sequence_data)
            data.extend(tokenized_data)

        # Convert tokenized data into PyTorch DataLoader
        input_ids_state = torch.stack([item["input_ids_state"] for item in data])
        attention_mask_state = torch.stack([item["attention_mask_state"] for item in data])
        input_ids_action = torch.stack([item["input_ids_action"] for item in data])
        attention_mask_action = torch.stack([item["attention_mask_action"] for item in data])
        input_ids_option = torch.stack([item["input_ids_option"] for item in data])
        attention_mask_option = torch.stack([item["attention_mask_option"] for item in data])
        rewards = torch.tensor([item["reward"] for item in data])

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(input_ids_state, attention_mask_state,
                                           input_ids_action, attention_mask_action,
                                           input_ids_option, attention_mask_option,
                                           rewards),
            batch_size=batch_size,
            shuffle=True
        )
        return data_loader

    def check_data_loader(self):
        # Load JSON files, collate data, and tokenize for model input
        data = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            json_data = self.load_json_data(file_path)
            sequence_data = self.collate_sequence_data(json_data)
            data_property = self.check_data(sequence_data)
            data.extend(data_property)
        return data


if __name__ == "__main__":
    # Example usage:
    folder_path = "./storage/v0.1"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    data_loader = DataLoader(folder_path, tokenizer)
    training_data_loader = data_loader.get_data_loader(batch_size=32)
    for batch in training_data_loader:
        # Access batched data here
        input_ids_state, attention_mask_state, input_ids_action, attention_mask_action, input_ids_option, attention_mask_option, rewards = batch
        # Your training loop or other processing here
        print(input_ids_state)