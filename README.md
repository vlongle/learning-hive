- Note on replay_buffer and transfer data. Results on data transfer is much worse than previously likely due to the difference between the new hash_data resevoir sampling vs the always-replace sampling errornously implemented before.
- Note on modmod: we turn off adaptation to prevent the weird behavior that modmod is worse than modular fedAvg due to training on past tasks.





## Why FL monolithic kinda peeks at the last epoch?

Artifact of the asynchronous system code. The `train` method has a `final` flag indicating whether 