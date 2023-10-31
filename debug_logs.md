Running

```
python experiments/debug_joint_agent.py --num_epochs 100 --comm_freq 10
```
and

```
python experiments/debug_joint_agent.py --num_epochs 100 --comm_freq 50
```
return very different results with the `comm_freq=50` gives much better acc. Obviously something wrong with the resume training.


Even the logging doesn't seem correct. Why epochs=11 keeps showing up.