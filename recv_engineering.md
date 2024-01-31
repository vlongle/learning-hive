
Different prefilter methods

-Oracle: 
    - couting OOD as 1.0: 88%
    - couting OOD as 0.0: 75.58%
- Prefilter with raw image feature (no thresholding): 32.3%
- No prefilter: 17.3% 

Corrected the accuracy calculation based on the entire sample
- Oracle: 85.17%
- prefilter with raw images: 32.3%
- No prefilter: 17.3%

Also, so far our prefiltering methods do not have a thresholding mechanism. We can try to add that in.

Oracle prefiltering in calculating acc (solve part of the OOD but not miscalibration between tasks)
- Oracle: 85.17% (same as before)
- prefilter with raw images: 45.64%
- No prefilter: 24.47%


__Mental check the oracle__
The com. graph is now disjoint with only self-loops: so the receiver is seeking data from itself. The val acc of a lot of these methods are ~97% so it's weird that the oracle is only 85%. self-loop yields about 57% because these are hard examples so the agent gets it wrong.

query (N,...)
M = number of available tasks at the sender
Compute the cluster distance matrix (N, M, num_classes_per_task=2)
where there are some strategies:
- distance to each class centroid
- avg distance to k closest neighbors from that class (K=1, or K=entire class size)

Then reduce to (N, M) by computing the variance |a-b|. 


Testing out some calibration / uncertainty metric:
(OOD, ID)
Entropy: 0.6544496814409891 0.6355584263801575
Least confidence: -0.6206618547439575 -0.6493780016899109
margin: -0.5019850730895996 -0.6327212254206339
This is WEIRDDDDD
random: 0.5076344609260559 0.5227058331171671

Normally, you'd expect random[ood] ~= random[id], but we just have more ood than id data
so the variance of id is higher. We should use some OOD detection method with some conservative bias towards classifying as OOD.

Maybe, just the problem with modular. Monolithic should NOT suffer from this problem because it always contrast against other stuff.



## Bug
To implement: if the pre-filter detect OOD for a sample, we need to send all zeros back. Or the receiver needs to use the prefilter-mask to only select the in-distribution samples from incoming data.


## TODOs
Another pre-filter option. Look at the "discrimination" between classes.

TODO: just viz the embedding 2D projection with OOD vs IID data.
- Uniform way to visualize
    - embedding for everything
    - viusalize the dist of OOD scores

## Ideas
- Eric: Train a global meta-model to do this embedding visualization.
- Me: send the prototype of the class as well (?)
- Do consistency-based stuff.





Contrastive.
Actually, we can do the 


### Thresholding

Either KNN or Local density

https://kunalbharadkar.medium.com/outlier-anomalies-detection-using-unsupervised-machine-learning-8a25e66de85c#:~:text=An%20absolute%20gem!-,Detecting%20outliers%20using%20KNN,and%20measures%20the%20average%20distance.



__results on the self stuff for tuning__

without separation training, task_acc=95.627534
    accuracy     0.594661
    precision    0.956400
    recall       0.520961
    f1           0.651315


test_lambda=1, task_acc=94.799746
    accuracy     0.677214
    precision    0.959260
    recall       0.618850
    f1           0.711905

test_lambda=2, task_acc=93.382346

using training set
    accuracy     0.727517
    precision    0.982794
    recall       0.678459
    f1           0.76904
using replay
    accuracy     0.705599
    precision    0.982619
    recall       0.653694
    f1           0.73981

test_lambda=5, task_acc=89.095391
using replay
    accuracy     0.710590
    precision    0.966659
    recall       0.664180
    f1           0.747241

test_lambda=10, task_acc=85.104223
using replay
    accuracy     0.677214
    precision    0.959260
    recall       0.618850
    f1           0.71190

__results on search quality__

without separation training,
- Oracle: 85.17% (same as before)
- prefilter with raw images: 45.64%
- No prefilter: 24.47%

with lambda=2
- Oracle: 78%
- prefilter with raw image: 31%, 40%
- No prefilter: 19%, 24%

TO DEBUG: wtf, why lambda doesn't improve the no_prefilter performance?

Open a notebook and debug the receiver KNN stuff,



Must be bug somewhere. Oracle got 80% which means that as long as the OOD stuff is fixed, the embedding search is very strong.

Just going to use the global labels.