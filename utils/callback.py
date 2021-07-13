class Checkpoint:
    def __init__(
        self,
        accuracy_threshold: float = 1.0,
        patience: int = 5,
    ) -> None:
        self._accuracy_threshold = accuracy_threshold
        self._patience = patience
        self._lookback = 0
        self._best_accuracy = None
        self._best_epoch = None
        self._best_training_time = None

    @property
    def best_accuracy(self):
        return self._best_accuracy

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_training_time(self):
        return self._best_training_time

    def create(
        self, accuracy: float,
        epoch: int,
        training_time: float,
    ) -> None:
        if self._best_accuracy is None or accuracy > self._best_accuracy:
            self._lookback = 0
            self._best_accuracy = accuracy
            self._best_epoch = epoch
            self._best_training_time = training_time
        else:
            self._lookback += 1

    def should_stop(self) -> bool:
        if self._lookback >= self._patience:
            return True
        elif self._best_accuracy >= self._accuracy_threshold:
            return True

        return False
