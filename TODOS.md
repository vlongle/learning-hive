- Modify grad to log to a different record.csv.
- Write IID few-shot learning eval, and look at forward transfer.
- Ensure that there's no bug in `analyze.py` (We need to now assume that that the `num_epochs` is the same 
everywhere).


## Debugs
`dropout=0.0`, update_modules takes a massive hit in accuracy of prev task. The loss is also too high to make sense. Loss increases but accuracy stays the same: is because of "overfitting" / miscalibration,
the model gets some stuff correctly but becomes increasingly wrong on some data points, making loss very high. Massive hit due to the fact that the replay buffer size is too small (data point = 32), combined with
this overfitting of dropout leads to a large decrease in average loss but also a massive decrease in accuracy.

Interestingly, the contrastive learning does not have this problem because the contrastion implicitly implements a sort of regularization (e.g.,
through data augmentation).

Our contrastive (dropout=0.0) took a hit during assimilation. (contrastive)
```
[2023-04-14 13:11:21,513][root][INFO] - epochs: 99, training task: 4
[2023-04-14 13:11:21,513][root][INFO] -         task: 0 loss: 1.274     acc: 0.734
[2023-04-14 13:11:21,514][root][INFO] -         task: 1 loss: 0.882     acc: 0.788
[2023-04-14 13:11:21,515][root][INFO] -         task: 2 loss: 0.865     acc: 0.824
[2023-04-14 13:11:21,515][root][INFO] -         task: 3 loss: 0.939     acc: 0.800
[2023-04-14 13:11:21,516][root][INFO] -         task: 4 loss: 0.996     acc: 0.794
[2023-04-14 13:11:21,517][root][INFO] -         task: avg       loss: 0.991     acc: 0.788
[2023-04-14 13:11:25,660][root][INFO] - epochs: 100, training task: 4
[2023-04-14 13:11:25,660][root][INFO] -         task: 0 loss: 1.330     acc: 0.724
[2023-04-14 13:11:25,661][root][INFO] -         task: 1 loss: 0.869     acc: 0.786
[2023-04-14 13:11:25,662][root][INFO] -         task: 2 loss: 0.936     acc: 0.816
[2023-04-14 13:11:25,662][root][INFO] -         task: 3 loss: 0.990     acc: 0.778
[2023-04-14 13:11:25,663][root][INFO] -         task: 4 loss: 1.035     acc: 0.774
[2023-04-14 13:11:25,664][root][INFO] -         task: avg       loss: 1.032     acc: 0.776
```

With dropout=0.5, (contrastive)
```
[2023-04-14 13:26:31,184][root][INFO] - epochs: 99, training task: 4
[2023-04-14 13:26:31,184][root][INFO] -         task: 0 loss: 1.058     acc: 0.558
[2023-04-14 13:26:31,185][root][INFO] -         task: 1 loss: 0.591     acc: 0.784
[2023-04-14 13:26:31,186][root][INFO] -         task: 2 loss: 0.633     acc: 0.732
[2023-04-14 13:26:31,186][root][INFO] -         task: 3 loss: 0.824     acc: 0.694
[2023-04-14 13:26:31,187][root][INFO] -         task: 4 loss: 1.632     acc: 0.744
[2023-04-14 13:26:31,187][root][INFO] -         task: avg       loss: 0.947     acc: 0.702
[2023-04-14 13:26:35,501][root][INFO] - epochs: 100, training task: 4
[2023-04-14 13:26:35,501][root][INFO] -         task: 0 loss: 1.084     acc: 0.554
[2023-04-14 13:26:35,502][root][INFO] -         task: 1 loss: 0.606     acc: 0.778
[2023-04-14 13:26:35,503][root][INFO] -         task: 2 loss: 0.609     acc: 0.744
[2023-04-14 13:26:35,504][root][INFO] -         task: 3 loss: 0.796     acc: 0.710
[2023-04-14 13:26:35,504][root][INFO] -         task: 4 loss: 1.489     acc: 0.768
[2023-04-14 13:26:35,505][root][INFO] -         task: avg       loss: 0.917     acc: 0.711
```

with dropout=0.5 turned on during `update_modules` (contrastive)
```
[2023-04-14 13:45:56,216][root][INFO] - epochs: 99, training task: 4
[2023-04-14 13:45:56,216][root][INFO] -         task: 0 loss: 1.354     acc: 0.728
[2023-04-14 13:45:56,217][root][INFO] -         task: 1 loss: 0.764     acc: 0.832
[2023-04-14 13:45:56,218][root][INFO] -         task: 2 loss: 0.720     acc: 0.822
[2023-04-14 13:45:56,218][root][INFO] -         task: 3 loss: 0.996     acc: 0.790
[2023-04-14 13:45:56,219][root][INFO] -         task: 4 loss: 0.992     acc: 0.790
[2023-04-14 13:45:56,220][root][INFO] -         task: avg       loss: 0.965     acc: 0.792
[2023-04-14 13:46:00,476][root][INFO] - epochs: 100, training task: 4
[2023-04-14 13:46:00,476][root][INFO] -         task: 0 loss: 1.475     acc: 0.694
[2023-04-14 13:46:00,477][root][INFO] -         task: 1 loss: 0.999     acc: 0.786
[2023-04-14 13:46:00,478][root][INFO] -         task: 2 loss: 0.833     acc: 0.796
[2023-04-14 13:46:00,478][root][INFO] -         task: 3 loss: 1.952     acc: 0.670
[2023-04-14 13:46:00,479][root][INFO] -         task: 4 loss: 12.329    acc: 0.212
[2023-04-14 13:46:00,479][root][INFO] -         task: avg       loss: 3.518     acc: 0.632
```

with `dropout=0.1` (contrastive)

```
[2023-04-14 15:39:57,375][root][INFO] - epochs: 99, training task: 4
[2023-04-14 15:39:57,376][root][INFO] -         task: 0 loss: 1.171     acc: 0.738
[2023-04-14 15:39:57,377][root][INFO] -         task: 1 loss: 0.710     acc: 0.820
[2023-04-14 15:39:57,378][root][INFO] -         task: 2 loss: 0.994     acc: 0.788
[2023-04-14 15:39:57,378][root][INFO] -         task: 3 loss: 0.996     acc: 0.804
[2023-04-14 15:39:57,379][root][INFO] -         task: 4 loss: 1.284     acc: 0.830
[2023-04-14 15:39:57,379][root][INFO] -         task: avg       loss: 1.031     acc: 0.796
[2023-04-14 15:40:02,894][root][INFO] - epochs: 100, training task: 4
[2023-04-14 15:40:02,894][root][INFO] -         task: 0 loss: 1.300     acc: 0.706
[2023-04-14 15:40:02,897][root][INFO] -         task: 1 loss: 0.733     acc: 0.824
[2023-04-14 15:40:02,898][root][INFO] -         task: 2 loss: 1.033     acc: 0.784
[2023-04-14 15:40:02,899][root][INFO] -         task: 3 loss: 1.368     acc: 0.750
[2023-04-14 15:40:02,901][root][INFO] -         task: 4 loss: 1.235     acc: 0.828
[2023-04-14 15:40:02,901][root][INFO] -         task: avg       loss: 1.134     acc: 0.778
```
dropout=0.25 (contrastive). Update modules consistently degrade the performance.

```
[2023-04-14 15:50:17,296][root][INFO] - epochs: 99, training task: 4
[2023-04-14 15:50:17,296][root][INFO] -         task: 0 loss: 0.927     acc: 0.734
[2023-04-14 15:50:17,297][root][INFO] -         task: 1 loss: 0.660     acc: 0.832
[2023-04-14 15:50:17,298][root][INFO] -         task: 2 loss: 0.699     acc: 0.790
[2023-04-14 15:50:17,299][root][INFO] -         task: 3 loss: 0.883     acc: 0.780
[2023-04-14 15:50:17,299][root][INFO] -         task: 4 loss: 1.135     acc: 0.846
[2023-04-14 15:50:17,300][root][INFO] -         task: avg       loss: 0.861     acc: 0.796
[2023-04-14 15:50:22,886][root][INFO] - epochs: 100, training task: 4
[2023-04-14 15:50:22,886][root][INFO] -         task: 0 loss: 0.948     acc: 0.730
[2023-04-14 15:50:22,888][root][INFO] -         task: 1 loss: 0.736     acc: 0.810
[2023-04-14 15:50:22,889][root][INFO] -         task: 2 loss: 0.791     acc: 0.772
[2023-04-14 15:50:22,891][root][INFO] -         task: 3 loss: 0.924     acc: 0.774
[2023-04-14 15:50:22,892][root][INFO] -         task: 4 loss: 1.112     acc: 0.836
[2023-04-14 15:50:22,893][root][INFO] -         task: avg       loss: 0.902     acc: 0.784
```

What if we allow the task specific decoder to change during `update_modules` step for contrastion?

dropout=0.0

```
[2023-04-14 16:04:52,610][root][INFO] - epochs: 99, training task: 4
[2023-04-14 16:04:52,610][root][INFO] -         task: 0 loss: 1.360     acc: 0.740
[2023-04-14 16:04:52,611][root][INFO] -         task: 1 loss: 0.727     acc: 0.812
[2023-04-14 16:04:52,612][root][INFO] -         task: 2 loss: 0.954     acc: 0.802
[2023-04-14 16:04:52,612][root][INFO] -         task: 3 loss: 1.270     acc: 0.772
[2023-04-14 16:04:52,613][root][INFO] -         task: 4 loss: 1.160     acc: 0.798
[2023-04-14 16:04:52,614][root][INFO] -         task: avg       loss: 1.094     acc: 0.785
[2023-04-14 16:04:58,084][root][INFO] - epochs: 100, training task: 4
[2023-04-14 16:04:58,084][root][INFO] -         task: 0 loss: 1.464     acc: 0.732
[2023-04-14 16:04:58,085][root][INFO] -         task: 1 loss: 0.749     acc: 0.796
[2023-04-14 16:04:58,086][root][INFO] -         task: 2 loss: 1.288     acc: 0.730
[2023-04-14 16:04:58,086][root][INFO] -         task: 3 loss: 1.183     acc: 0.776
[2023-04-14 16:04:58,087][root][INFO] -         task: 4 loss: 1.126     acc: 0.820
[2023-04-14 16:04:58,088][root][INFO] -         task: avg       loss: 1.162     acc: 0.771
```

What if we just do `component_update_freq` more often? Doesn't seem to work either.

```
[2023-04-14 16:17:16,814][root][INFO] - epochs: 99, training task: 4
[2023-04-14 16:17:16,814][root][INFO] -         task: 0 loss: 1.360     acc: 0.740
[2023-04-14 16:17:16,815][root][INFO] -         task: 1 loss: 0.727     acc: 0.812
[2023-04-14 16:17:16,816][root][INFO] -         task: 2 loss: 0.954     acc: 0.802
[2023-04-14 16:17:16,817][root][INFO] -         task: 3 loss: 1.270     acc: 0.772
[2023-04-14 16:17:16,817][root][INFO] -         task: 4 loss: 1.160     acc: 0.798
[2023-04-14 16:17:16,818][root][INFO] -         task: avg       loss: 1.094     acc: 0.785
[2023-04-14 16:17:22,235][root][INFO] - epochs: 100, training task: 4
[2023-04-14 16:17:22,235][root][INFO] -         task: 0 loss: 1.492     acc: 0.734
[2023-04-14 16:17:22,237][root][INFO] -         task: 1 loss: 0.736     acc: 0.798
[2023-04-14 16:17:22,237][root][INFO] -         task: 2 loss: 1.264     acc: 0.732
[2023-04-14 16:17:22,238][root][INFO] -         task: 3 loss: 1.204     acc: 0.776
[2023-04-14 16:17:22,238][root][INFO] -         task: 4 loss: 1.118     acc: 0.810
[2023-04-14 16:17:22,239][root][INFO] -         task: avg       loss: 1.163     acc: 0.770
```

It might be the artifact of different twoCrop transforms that we're using. If we're using contrastive, we should also store
the other twoCrops to stabilize training! Because different transformation (strong augmentation) would lead to very large loss during
the `update_modules` part, leading to the network having to significantly changes the modules' weights, which are NOT good for stability.

Keeping same transforms, dropout=0.0, and one update_modules step.

```
[2023-04-14 17:09:02,809][root][INFO] - epochs: 99, training task: 4
[2023-04-14 17:09:02,809][root][INFO] -         task: 0 loss: 1.609     acc: 0.734
[2023-04-14 17:09:02,811][root][INFO] -         task: 1 loss: 0.754     acc: 0.830
[2023-04-14 17:09:02,811][root][INFO] -         task: 2 loss: 0.951     acc: 0.806
[2023-04-14 17:09:02,812][root][INFO] -         task: 3 loss: 1.158     acc: 0.788
[2023-04-14 17:09:02,813][root][INFO] -         task: 4 loss: 1.088     acc: 0.802
[2023-04-14 17:09:02,813][root][INFO] -         task: avg       loss: 1.112     acc: 0.792
Updating modules...
[2023-04-14 17:09:08,251][root][INFO] - epochs: 100, training task: 4
[2023-04-14 17:09:08,251][root][INFO] -         task: 0 loss: 1.730     acc: 0.710
[2023-04-14 17:09:08,253][root][INFO] -         task: 1 loss: 0.814     acc: 0.822
[2023-04-14 17:09:08,255][root][INFO] -         task: 2 loss: 0.989     acc: 0.796
[2023-04-14 17:09:08,256][root][INFO] -         task: 3 loss: 1.288     acc: 0.750
[2023-04-14 17:09:08,257][root][INFO] -         task: 4 loss: 1.019     acc: 0.802
[2023-04-14 17:09:08,258][root][INFO] -         task: avg       loss: 1.168     acc: 0.776
```


sanity check: (same crop)

```
[2023-04-14 17:25:00,192][root][INFO] - epochs: 99, training task: 4
[2023-04-14 17:25:00,192][root][INFO] -         task: 0 loss: 1.360     acc: 0.740
[2023-04-14 17:25:00,193][root][INFO] -         task: 1 loss: 0.727     acc: 0.812
[2023-04-14 17:25:00,194][root][INFO] -         task: 2 loss: 0.954     acc: 0.802
[2023-04-14 17:25:00,195][root][INFO] -         task: 3 loss: 1.270     acc: 0.772
[2023-04-14 17:25:00,195][root][INFO] -         task: 4 loss: 1.160     acc: 0.798
[2023-04-14 17:25:00,196][root][INFO] -         task: avg       loss: 1.094     acc: 0.785
Updating modules...
loss:  0.566232442855835
loss:  0.4228939712047577
loss:  0.5333210229873657
loss:  0.5533656477928162
loss:  0.4291853606700897
loss:  0.3885042071342468
loss:  0.7677289843559265
loss:  1.1621326208114624
loss:  0.7519549131393433
loss:  0.4300956726074219
loss:  0.5338665843009949
loss:  0.5707424879074097
loss:  0.5472257137298584
loss:  0.5324690341949463
loss:  0.5294522643089294
loss:  0.5367009043693542
loss:  0.5746378898620605
loss:  0.4188630282878876
loss:  0.4552871286869049
loss:  0.4145170748233795
loss:  0.5748213529586792
loss:  0.7914441823959351
[2023-04-14 17:25:05,566][root][INFO] - epochs: 100, training task: 4
[2023-04-14 17:25:05,566][root][INFO] -         task: 0 loss: 1.539     acc: 0.728
[2023-04-14 17:25:05,568][root][INFO] -         task: 1 loss: 0.854     acc: 0.782
[2023-04-14 17:25:05,568][root][INFO] -         task: 2 loss: 1.365     acc: 0.746
[2023-04-14 17:25:05,569][root][INFO] -         task: 3 loss: 1.222     acc: 0.782
[2023-04-14 17:25:05,569][root][INFO] -         task: 4 loss: 1.095     acc: 0.816
[2023-04-14 17:25:05,570][root][INFO] -         task: avg       loss: 1.215     acc: 0.771
```


sanity check: (different crop)


Updating modules...
loss:  0.534174919128418
loss:  0.4035310447216034
loss:  0.5433964133262634
loss:  0.5352720022201538
loss:  0.3985607624053955
loss:  0.39676591753959656
loss:  0.7035078406333923
loss:  1.0894681215286255
loss:  0.7261749505996704
loss:  0.41712138056755066
loss:  0.5117488503456116
loss:  0.5144395232200623
loss:  0.5251771211624146
loss:  0.5199273824691772
loss:  0.5402238368988037
loss:  0.5264296531677246
loss:  0.5152176022529602
loss:  0.4139736592769623
loss:  0.4519527852535248
loss:  0.38200899958610535
loss:  0.5559605360031128
loss:  0.7633329629898071


__BUGS__


loss:  0.5423436760902405
task tensor(0) size 1 no components 5 cl: tensor(3.3379e-06, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.4722, device='cuda:0')
task tensor(1) size 1 no components 5 cl: tensor(5.2849e-06, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.1534, device='cuda:0')
task tensor(2) size 3 no components 5 cl: tensor(1.0372, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0266, device='cuda:0')
task tensor(3) size 2 no components 5 cl: tensor(0.4320, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.3231, device='cuda:0')
task tensor(4) size 57 no components 5 cl: tensor(4.1791, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.5203, device='cuda:0')
mode: None both

For some reasons, cl (normalized already by the no. of points) for the current task is just much larger than the previous task.

Before update_modules

{0: 0.734, 1: 0.83, 2: 0.806, 3: 0.788, 4: 0.798, 'avg': 0.7912000000000001}

After update_modules

{0: 0.71, 1: 0.776, 2: 0.818, 3: 0.766, 4: 0.782, 'avg': 0.7704}


During update structures, cl is normal..

task 4 size 64 no components 5 cl: tensor(0.0697, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0174, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 4 cl: tensor(0.0757, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0187, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 5 cl: tensor(0.0663, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0176, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 4 cl: tensor(0.0749, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0196, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 5 cl: tensor(0.0673, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0173, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 4 cl: tensor(0.0795, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0197, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 5 cl: tensor(0.0657, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0181, device='cuda:0', grad_fn=<DivBackward0>)
task 4 size 64 no components 4 cl: tensor(0.0782, device='cuda:0', grad_fn=<DivBackward0>) ce: tensor(0.0202, device='cuda:0', grad_fn=<DivBackward0>)


Observations:
- should probably free up the projectors during `update_modules`
- still doesn't explain why cl loss is so large compared to when `update_structure` is called. Even `ce` loss is also large compared to `update_structure` logging.


Visualize the actual images...

Ok, looks normal.

It's the loss reduction BS that Jorge has in his code, WTF????
`gradient_step` does NOT 


The reduction of the supcontrast loss is actually buggy!!


Very problematic... Need to do the math and think through this...


After task balancing sampler:

before `update_modules`: {0: 0.734, 1: 0.83, 2: 0.806, 3: 0.788, 4: 0.798, 'avg': 0.7912000000000001}
after `update_modules`: {0: 0.668, 1: 0.78, 2: 0.784, 3: 0.77, 4: 0.728, 'avg': 0.7460000000000001}

After unfreezing projector head as well.
{0: 0.682, 1: 0.812, 2: 0.748, 3: 0.748, 4: 0.798, 'avg': 0.7576}

Fixing the normalization bug in Jorge's code.

{0: 0.702, 1: 0.81, 2: 0.754, 3: 0.752, 4: 0.792, 'avg': 0.7619999999999999}

Unfreezing both projector and decoder
{0: 0.688, 1: 0.804, 2: 0.764, 3: 0.726, 4: 0.786, 'avg': 0.7536}

Revert to before the `bug` (i.e. no task balancing and also no changing projection)

{0: 0.718, 1: 0.794, 2: 0.82, 3: 0.772, 4: 0.808, 'avg': 0.7824}


BEFORE:
{0: 0.734, 1: 0.83, 2: 0.806, 3: 0.788, 4: 0.798, 'avg': 0.7912000000000001}

Anyway, seems like `update_modules` step isn't worth it for contrastive with such a small number of step size..



## BUGS

Modular contrastive, final acc is much lower than avg_acc??? Why???