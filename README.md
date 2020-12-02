# Object-oriented rule-based agent for Google Research Football

This agent is currently at top 20 and expecting to get a silver medal.
You can browse previous versions. There are 65 versions in total.
The last version v100 is actually refactored v039, 
which is the most consistently good agent so far.
You can clone specific commits and use all these 65 agents.
They can create a diverse training gym for a RL agent.

### Environment Setup
`run install.sh`

### Evaluation against a benchmark agent
`python evaluate.py --benchmark /kaggle_simulations/agent/public_benchmark.py --num-games 16`
