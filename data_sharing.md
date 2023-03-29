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
Post(odd): pass
