# Data Sharing
## Sender
Candidate pool: 
- Current tasks.
- all the training tasks ✅
"Backward transfer". Agent A has some data D(A, t) at time t, which is not useful at the time to B. But at t + h, D(A, t) becomes useful to B.

Dedup protocol:
- Receiver sends back feedback and indicator of which data points are kept.
- Sender remembers that per receiver, and never consider those data points again.


Recommendation engine.
Freeze the modules, and per agent and per task, finetune a regressor head downstream.
Exploration: epsilon with geometric decay or some aggressive schedule.


### Communication round

Pre (parallel) -> com (sequential) -> post (parallel)

- Even round: sending data.
- Odd round: sending feedback

Pre(even): greedy eps compute data
Com(even): send data
Post(even): evaluate data -> feedback.


Pre(odd): pass
Com(odd): send feedback
Post(odd): update engine & eps.


## Receiver

### Communication Round

- Even round: sending query.
- Odd round: sending data.

Pre(even): compute query from my validation sets.
Com(even): send query
Post(even): pass

Pre(odd): compute data from query using image search + outlier removal.
Com(odd): send data
Post(odd): learn from data


__Learn from data__
Receiver sends back (x, y) where y is their own labels, and the task_id `t` attached to which query it is from.
If contrastive learning, then use all the shared data on `t`.
If not, then we have to pretend the majority labels (another heuristic is to use the closest labels) is the true labels, and learn using only those points on `t`.

__Image Search__
Upon receiving a query, we get k-nearest neighbors using the cosine distance of the embeddings. Then, we use the distance to determine if
they are outliers using in-training data.



All data sharing should implement a dedup routine!