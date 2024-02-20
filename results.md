## Vanilla

=====FINAL ACC======
algo        dataset       use_contrastive
modular     cifar100      False              78.783750
                          True               76.743750
            fashionmnist  False              93.083125
                          True               93.697500
            kmnist        False              81.855625
                          True               84.472500
            mnist         False              94.623223
                          True               95.624038
monolithic  cifar100      False              58.748750
                          True               57.881250
            fashionmnist  False              92.138125
                          True               92.776875
            kmnist        False              79.009375
                          True               83.340625
            mnist         False              90.331536
                          True               94.768338


=====AVG ACC======
algo        dataset       use_contrastive
modular     cifar100      False              78.529979
                          True               76.345013
            fashionmnist  False              93.593048
                          True               93.654985
            kmnist        False              82.389468
                          True               85.442454
            mnist         False              94.787107
                          True               95.746557
monolithic  cifar100      False              63.623694
                          True               62.402608
            fashionmnist  False              92.606829
                          True               93.041261
            kmnist        False              79.552352
                          True               82.989010
            mnist         False              91.514573
                          True               94.860344


at start_epoch = 21

=====FORWARD======
algo        dataset       use_contrastive
modular     fashionmnist  True               70.486607
            kmnist        True               61.055357
            mnist         True               68.606603
monolithic  fashionmnist  True               89.035714
            kmnist        True               79.141964
            mnist         True               89.783732
Name: forward, dtype: float64




## Grad sync
=====FINAL ACC======
algo        dataset       use_contrastive
modular     fashionmnist  True               93.580671
            kmnist        True               83.862269
            mnist         True               95.071818
monolithic  fashionmnist  True               93.152199
            kmnist        True               82.286921
            mnist         True               94.927760

=====AVG ACC======
algo        dataset       use_contrastive
modular     fashionmnist  True               93.701065
            kmnist        True               85.178785
            mnist         True               95.674834
monolithic  fashionmnist  True               93.268908
            kmnist        True               82.987226
            mnist         True               94.914154



start_epoch = 21

=====FORWARD======
algo        dataset       use_contrastive
modular     fashionmnist  True               76.320139
            kmnist        True               61.408102
            mnist         True               67.448571
monolithic  fashionmnist  True               90.245139
            kmnist        True               78.157870
            mnist         True               89.850139
Name: forward, dtype: float64



algo     dataset       use_contrastive
modular  fashionmnist  True               93.790394
         kmnist        True               84.109259
         mnist         True               95.334392


## Turning Modular

result_dir = "vanilla_results"
    dataset        algo  use_contrastive    seed agent_id   avg_acc  final_acc   forward  backward  catastrophic
0  cifar100     modular            False  seed_0  agent_1  0.775173     0.7875  0.196471  0.017853          0.59
0  cifar100     modular            False  seed_0  agent_2  0.767970     0.7801  0.199294  0.016905          0.12
0  cifar100     modular            False  seed_0  agent_6  0.810245     0.7995  0.213412  0.020916          0.20
0  cifar100     modular            False  seed_0  agent_7  0.794568     0.7871  0.205647  0.018768          0.62
0  cifar100     modular            False  seed_0  agent_4  0.789978     0.7736  0.202824  0.020189          0.87
0  cifar100     modular            False  seed_0  agent_3  0.764879     0.7798  0.188235  0.018221         -0.13
0  cifar100     modular            False  seed_0  agent_0  0.780671     0.7942  0.200824  0.017463          0.20
0  cifar100     modular            False  seed_0  agent_5  0.798915     0.8009  0.199647  0.018989         -0.16

0  cifar100     modular             True  seed_0  agent_1  0.740929     0.7587  0.201647  0.016389          0.11
0  cifar100     modular             True  seed_0  agent_2  0.749622     0.7680  0.198471  0.015368          0.56
0  cifar100     modular             True  seed_0  agent_6  0.796976     0.7899  0.194941  0.019137          0.00
0  cifar100     modular             True  seed_0  agent_7  0.770605     0.7604  0.202588  0.017379          1.42
0  cifar100     modular             True  seed_0  agent_4  0.769633     0.7452  0.188588  0.018663          1.23
0  cifar100     modular             True  seed_0  agent_3  0.732907     0.7624  0.211412  0.016368          0.25
0  cifar100     modular             True  seed_0  agent_0  0.775489     0.7765  0.204824  0.017105          1.01
0  cifar100     modular             True  seed_0  agent_5  0.771440     0.7784  0.217765  0.018653          0.08




result_dir = "cifar_lasttry_im_done_projector_no_freeze_scaling_2.0_temp_0.06_results"


    dataset     algo  use_contrastive    seed agent_id   avg_acc  final_acc   forward  backward  catastrophic
0  cifar100  modular             True  seed_0  agent_1  0.734576     0.7503  0.207882  0.015095          1.18
0  cifar100  modular             True  seed_0  agent_2  0.730013     0.7446  0.195882  0.013968          1.73
0  cifar100  modular             True  seed_0  agent_6  0.785582     0.7712  0.205412  0.018495          0.76
0  cifar100  modular             True  seed_0  agent_7  0.766055     0.7561  0.209176  0.016905          1.53
0  cifar100  modular             True  seed_0  agent_4  0.773231     0.7484  0.209765  0.021705          0.41
0  cifar100  modular             True  seed_0  agent_3  0.738278     0.7633  0.216118  0.016284          0.07
0  cifar100  modular             True  seed_0  agent_0  0.764395     0.7725  0.193529  0.017189          0.22
0  cifar100  modular             True  seed_0  agent_5  0.766867     0.7660  0.197882  0.017747          0.94



## TODO

Actually compute the number of modules for each methods!




## New Results

Baseline


=====FINAL ACC======
algo     dataset       use_contrastive
modular  fashionmnist  True               93.861719
         kmnist        True               84.894922
         mnist         True               95.621633
Name: final_acc, dtype: float64


monolithic  fashionmnist  
                          True               92.608906
            kmnist        
                          True               82.660078
            mnist         
                          True               95.240861
Name: final_acc, dtype: float64




No sync base

random

=====FINAL ACC======
algo     dataset       use_contrastive
modular  fashionmnist  True               93.298516
         kmnist        True               83.431484
         mnist         True               94.460275
Name: final_acc, dtype: float64



No random

=====FINAL ACC======
algo     dataset       use_contrastive
modular  fashionmnist  True               93.735859
         kmnist        True               85.038672
         mnist         True               95.332292
Name: final_acc, dtype: float64


Sync base

Random
=====FINAL ACC======
algo     dataset       use_contrastive
modular  fashionmnist  True               93.535781
         kmnist        True               83.325938
         mnist         True               94.257971
Name: final_acc, dtype: float64


No random
=====FINAL ACC======
algo     dataset       use_contrastive
modular  fashionmnist  True               93.843616
         kmnist        True               84.648281
         mnist         True               95.265822
Name: final_acc, dtype: float64



Receiver

=====FINAL ACC======
algo        dataset       use_contrastive
monolithic  fashionmnist  True               92.786406
            kmnist        True               82.811563
            mnist         True               95.029952
Name: final_acc, dtype: float64


Current: Something is really wrong... (BUG)
=====FINAL ACC======
algo     dataset       use_contrastive
modular  fashionmnist  True               60.190000
         kmnist        True               53.841328
         mnist         True               56.127706
Name: final_acc, dtype: float64





Recv all for the monolithic is taking a lot of time to run for some reasons?



TODO:
need to control for randomness.