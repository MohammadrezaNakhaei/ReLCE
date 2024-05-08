# ReLCE: Residual Learning and Context Encoding for Adaptive Offline-to-Online RL

## Instructions
Install d4rl follwing this [repo](https://github.com/Farama-Foundation/D4RL):

Install dependencies from requirements.txt
```sh
pip install -r rquirement.txt
```

Train the offline agent:
``` sh
python train_offline.py --task hopper-medium-v2
```

Train residual agent:
``` sh
python train.py --offline-path ./log/...
```

## Citation
```bibtex
@article{ReLCE2024,
    title={ Residual Learning and Context Encoding for Adaptive Offline-to-Online Reinforcement Learning},
    author={Mohammadreza Nakhaei, Aidan Scannell, Joni Pajarinen}
    journal={},
    year={2024}
}
```
