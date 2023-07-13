# Open Hearthstone
The OpenHearthstone is designed to be a robust data collection standard as well as a reinforcement learning incubator for research purposes. 

## Roadmap

The major roadmap for this repository is as follows.

- Data Standard Specification & Data S/L in `C#` and `Python`
- Data Collection with basic settings
- Implement a baseline RL-based deep learning network
  - Pretraining on text materials in game
  - Offline fully-supervised learning with human annotation
  - Online contrastive learning for continuous improvements

## Data Collection Specifications
### Game Saving
Core components for data storage is organized by game stored in separate files. For each game object, we use json structure to save the critical decision for each step.

```json
{
  "result": 1 (Win) / 0 (Tie) / -1 (Lose),
  "metadata": {
    // Metadata for the game
    "elapsed": 0,  // Elapsed time
    "turn": 0,  // Number of turns
    "action": 0,  // Number of actions
    ...
  },
  "sequence": {
    // List of state, action and option pairs
    "state": [
      [
        // List of entities
        {
          "card_id": "",
          "map": [
            "TAG": "VALUE"
          ]
        },
      ],
    ],
    "action": [
      [
        "from": 0,
        "to": 0,
        "type": 0,
        "choice": 0
      ],
    ],
    "option": [
      [
        // All end effector options
        [
          "from": 0,
          "to": 0,
          "type": 0,
          "choice": 0,
        ],
      ],
    ]
  }
}
```

### Data Definition

## Baseline Reinforcement Learning Pipeline Design

### Pretraining
```mermaid
graph LR;

    Text[In-game Text] ==> GPT([Generative Pretrained Transformers]);
    GPT -.-> Text;
```

### Offline Supervised Learning
```mermaid
graph TD;
    State ==> Policy;
    Predictor -.-> State;
    Policy --o Reward{{Reward}};
    Policy([Policy Network]) --o Action{{Action}};
    
    Reward <--> Result[[Game Result]];
    Action <--> Label[[Human Annotation]];
    Action <--> Option[[All Available Options]];
    State --> Predictor([Prediction Network]);
    Action --> Predictor;
```

### Online Contrastive Learning

```mermaid
graph TD;
    State ==> Policy;
    Predictor -.-> State;
    Policy --o Reward{{Reward}};
    Policy([Policy Network]) --o Action{{Action}};
    
    Reward <--> Result[[Pending Game Result]];
    Action <--> Label[[Contrastive Learning]];
    Action <--> Option[[All Available Options]];
    State --> Predictor([Prediction Network]);
    Action --> Predictor;
```
