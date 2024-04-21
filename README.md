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


## TODO:
- [x] Only FedAvg on modules (and not task-specific structures)
- [] implement FedAvg with momentum (see Flower module)
- [] check ray.put to see if we need deepcopy
- [] use fixed projections across all agents but still random across tasks?

Need to fix the bug in FedProx. Should only add L2 on the shared params and nothing else. Modular FedAvg and Modular FedProx should be the same.

Using contrastive loss primarily and does not allow cross entropy gradients to flow
back through the encoding.

## Notes
For remap_labels=True, modular will not work if we train with a lot of init_tasks because during assimilation
we don't change the components, and these components already learn to map labels to some previous labels. Encountering
new labels without changing the modules (but only re-restructuring as in modular as opposed to monolithic) means that 
the model will just not learn.
Solutions:
1. Either do remap_labels=False as before in the original paper (this might limit transfer across tasks though).
2. Reduce the number of init_tasks and original modules to limit overfitting. It doesn't really solve the root of the problem.
][root][INFO] -         task: 3 loss: 0.18648204755543465       acc: 0.9266331658291457
[2023-03-22 16:36:53,891][root][INFO] -         task: 4 loss: 0.17842456090797498       acc: 0.9373786407766991
[2023-03-22 16:36:53,892][root][INFO] -         task: 5 loss: 0.3132778021653727        acc: 0.8885616102110947
[2023-03-22 16:36:53,892][root][INFO] -         task: 6 loss: 0.09932567352207437       acc: 0.9765886287625418
[2023-03-22 16:36:53,893][root][INFO] -         task: 7 loss: 0.46160813278435825       acc: 0.8252134605725766
[2023-03-22 16:36:53,893][root][INFO] -         task: 8 loss: 0.4760802147345016        acc: 0.8140407288317256
[2023-03-22 16:36:53,894][root][INFO] -         task: 9 loss: 0.10803565069026523       acc: 0.9919697685403873
[2023-03-22 16:36:53,895][root][INFO] -         task: avg       loss: 0.21408524074298851       acc: 0.9248043895895405
[2023-03-22 16:36:53,904][root][INFO] - W/update: 0.99, WO/update: 0.99
[2023-03-22 16:36:53,904][root][INFO] - Not keeping new module. Total: 8
[2023-03-22 16:36:54,594][root][INFO] - epochs: 201, training task: 9
[2023-03-22 16:36:54,594][root][INFO] -         task: 0 loss: 0.11923505289038432       acc: 0.9567010309278351
[2023-03-22 16:36:54,595][root][INFO] -         task: 1 loss: 0.027654294873049757      acc: 0.9914974019839395
[2023-03-22 16:36:54,596][root][INFO] -         task: 2 loss: 0.17072897730646908       acc: 0.9394594594594594
[2023-03-22 16:36:54,596][root][INFO] -         task: 3 loss: 0.18648204755543465       acc: 0.9266331658291457
[2023-03-22 16:36:54,597][root][INFO] -         task: 4 loss: 0.17842456090797498       acc: 0.9373786407766991
[2023-03-22 16:36:54,598][root][INFO] -         task: 5 loss: 0.3132778021653727        acc: 0.8885616102110947
[2023-03-22 16:36:54,598][root][INFO] -         task: 6 loss: 0.09932567352207437       acc: 0.9765886287625418
[2023-03-22 16:36:54,599][root][INFO] -         task: 7 loss: 0.46160813278435825       acc: 0.8252134605725766
[2023-03-22 16:36:54,600][root][INFO] -         task: 8 loss: 0.4760802147345016        acc: 0.8140407288317256
[2023-03-22 16:36:54,600][root][INFO] -         task: 9 loss: 0.10803565069026523       acc: 0.9891355692017005
[2023-03-22 16:36:54,601][root][INFO] -         task: avg       loss: 0.21408524074298851       acc: 0.9245209696556718
[2023-03-22 16:36:54,642][root][INFO] - final components: 8
Results saved in less_overfit_results/ Also typically for remap_labels=True, we have to train much longer like 200 epochs.
3. Train representation learning through supcontrast (completely label-agnostic) and do prototype classifier downstream, or prevent
gradient from the entropy to affect the representation!
NOTE: this is currently NOT working for some reasons, maybe need the projector??
Still running into the pairwise contrastive bias problem...


The random structures across tasks might be problematic for aligning these representations.
- TODO: pick all ones and keep them fixed.


How are we not losing info. by just using some random linear transformation like this? MNIST are mostly black and white so it's a hail-mary judgement?


## Experiments Logs
Comparing modular vs monolithic vs contrastive.
on full and reduced datasets.



Recommendation stuff: contrastive objectives might be a bit terrible.



__Should not use random projection per task__: define the purpose of forward transfer. At least, we'll need a task-specific 
adapter before for it to work. Note that the `soft layer ordering` paper is about multi-task learning so it worked out for them.
See random_projection.py, with the same projection, it takes the 4th task 1 epoch to reach back 90%. With random projection per task,
it takes 4 epochs (Mono). (Mod): same projection takes 1 epoch to get back to 92%. The model didn't make any new components (keep components=3). With random per task projection, Mod took 5 epochs to get 90%, and have to use a new module. Generalization across tasks is non-existent!





TODO: multi-agent coordination is all over the place...
We should **not** parallelize the sending part because the receiver must be present at the end of the pipe when the caller makes the call.




FashionMNIST data augmentation

transforms.RandomResizedCrop(
                    size=self.net.i_size[0], scale=(0.2, 1.), antialias=True),
transforms.RandomHorizontalFlip(),
__maybe__ also salt pepper (gaussian) noise like: https://github.com/1202kbs/EBCLR/blob/main/configs_ebclr/fmnist_b64.yaml?


train for 100 epochs.
modular     fashionmnist  False              0.930831
                          True               0.925538

increase to 150 epochs
modular     fashionmnist  False              0.940206
                          True               0.935081


Use 
transforms.RandomHorizontalFlip(),
transforms.GaussianBlur(
                        kernel_size=3, sigma=(0.1, 2.0)),
transforms.Lambda(
                lambda x: x + torch.randn(x.size()).to(self.net.device) * 0.05),
-> modular     fashionmnist  True               0.936975

Changing contrast_mode to "all" for curiosity.



Cifar100.
Why does the results for cifar100 monolithic False is so high??
(Diff: replacement=True, one encoder for everything)
algo        dataset   use_contrastive
modular     cifar100  False              0.78525
                      True               0.73130
monolithic  cifar100  False              0.73740
                      True               0.74390

Jorge paper: no comp. should be around 51.6%
Our shows 73%.

Rerunning with replacement=False.
God, plz no, maybe have to do different encoders stupid trick again.



### Bugs

## CIFAR SHIT

Stuff to try, maybe even freeze the projector.

__invasive modification__: do a lot of backward update on the projector ONLY (and NOT the 
structure), one hacky way to do this is to concat megadataset to trainset during normal dynamic training.

Debug why contrastive is using so few components.

Visualize the accuracy over time.


    dataset     algo  use_contrastive    seed agent_id   avg_acc  final_acc   forward  backward  catastrophic
0  cifar100  modular             True  seed_0  agent_1  0.786419     0.7828  0.587529  0.008353          1.09
0  cifar100  modular             True  seed_0  agent_2  0.778686     0.7729  0.658471  0.007807          1.01
0  cifar100  modular             True  seed_0  agent_6  0.809990     0.7875  0.645294  0.008492          1.78
0  cifar100  modular             True  seed_0  agent_7  0.800894     0.7753  0.648235  0.008674          1.30
0  cifar100  modular             True  seed_0  agent_4  0.794289     0.7610  0.621412  0.008332          2.00
0  cifar100  modular             True  seed_0  agent_3  0.781079     0.7869  0.622353  0.009155          0.35
0  cifar100  modular             True  seed_0  agent_0  0.787815     0.7741  0.617059  0.007626          1.69
0  cifar100  modular             True  seed_0  agent_5  0.794106     0.7884  0.629176  0.009594          0.28



0  cifar100     modular            False  seed_0  agent_1  0.774651     0.7837  0.715176  0.017884          0.05
0  cifar100     modular            False  seed_0  agent_2  0.768384     0.7783  0.715529  0.017011          0.13
0  cifar100     modular            False  seed_0  agent_6  0.808626     0.7995  0.715412  0.020621         -0.84
0  cifar100     modular            False  seed_0  agent_7  0.793669     0.7871  0.713176  0.018579         -0.12
0  cifar100     modular            False  seed_0  agent_4  0.789613     0.7736  0.691765  0.020095          0.70
0  cifar100     modular            False  seed_0  agent_3  0.764018     0.7798  0.696941  0.018021         -0.44
0  cifar100     modular            False  seed_0  agent_0  0.780143     0.7917  0.711059  0.017463         -0.12
0  cifar100     modular            False  seed_0  agent_5  0.798304     0.8009  0.699765  0.018853         -0.38


There's a bug somewhere in our code!! No dropout for cifar is significantly WORSE than normal.



## NOTE TO FL STUFF

One thing that happens with modular is the previous structure is not optimized again, and so they became fcked.



## Interesting Note

0.9713408393039918
with training the contrastive stuff alone (mode='cl') and use a random decoder, we still get 97% accuracy on MNIST.

with mode='both' (contrast + cross entropy)
0.9790174002047083

with mode='ce_finetune' (only train the last layer keeping the random encoder fixed)
0.5015353121801432





