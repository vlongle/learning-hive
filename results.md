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