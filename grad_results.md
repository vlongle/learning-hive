Averaging for monolithic method doesn't seem to work because it totally destroys the learning, and the ER is not big enough to relearn this.

- Use fisher information (EWC/momentum encoder) to mitigate the averaging.
    - by chance, two networks offload two independent knowledge into the same weight part.
- Use some sort of MAML / meta-learning although I actually don't quite understand the dynamic of meta learning. This
seems to be a "inverse meta learning problem"???