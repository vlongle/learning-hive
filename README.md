# learning-hive

Easy datasets: MNIST, FashionMNIST
Harder dataset: KMNIST, CIFAR100

## Sharing Data
To figure out whether we should learn from labels or not. And whether to align labels...

## Sharing gradients
MonoGrad would probably works for Easy-Data (more zero-shot
generalization than Mono) but not Hard-Data because
using one model to learn all is too difficult (basically increase the no. 
of tasks that a model has to learn)

ModularGrad can probably maintain the previous performance as in Modular because we
are allowed dynamic modules. We again can obtain zero-shot improvement on short-sequence easy dataset.
For longer-sequence Hard-Data, the local flexibility of ModularGrad allows us to learn.

## Sharing Modules
In longer-sequence Hard-Data, task distribution can overlap still, which create opportunities for sharing modules.