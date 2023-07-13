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


## Baseline Reinforcement Learning Pipeline Design

```mermaid
graph LR;
title Baseline RL Pipeline Diagram

Section Pretraining
    In-game Text ==> Generative Pretrained Transformers (GPT);
    Generative Pretrained Transformers (GPT) -.-> In-game Text;
Section Offline Supervised Training
    State ==> Policy;
    Policy --o Action;
    Policy --o Reward;
    Reward <--> Game Result;
    Action <--> Human Annotation;
    Action <--> All Available Options;

    State ==> Predictor;
    Action ==> Predictor;
    Predictor --o State;
Section Online Contrastive Learning
  State ==> Policy;
    Policy --o Action;
    Policy --o Reward;
    Reward <--> Contrastive Learning based on Game Result + Curiosity based Exploration;
    Action <--> All Available Options;

    State ==> Predictor;
    Action ==> Predictor;
    Predictor --o State;
```
