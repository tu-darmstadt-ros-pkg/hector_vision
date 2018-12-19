
class CancellationToken:
    def __init__(self):
        self.cancellation_requested = False

    def cancel(self):
        self.cancellation_requested = True
