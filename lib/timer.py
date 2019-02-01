# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 10: # don't use warm_up
            self.warm_up += 1
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff

class Timers():
    ''' A collect of Timer()
    Usage:
        see __main__
    '''
    def __init__(self):
        self.timers = {}

    def __len__(self):
        return len(self.timers)

    def __getitem__(self, key):
        if key not in self.timers:
            self.timers[key] = Timer()
        return self.timers[key]

    def print(self):
        print ('@@@@ Timer Report ===>')
        for key, timer in self.timers.items():
            print ('@@@@ [%-15s]: aver %10.3fs on %10d hits'\
                        %(key, timer.average_time, timer.calls))

if __name__ == '__main__':
    timers = Timers()

    for _ in range(1,10):
        timers['tag1'].tic()
        time.sleep(0.1)
        timers['tag1'].toc()

    timers['tag2'].tic()
    for _ in range(1,10):
        time.sleep(0.1)
    timers['tag2'].toc()

    timers.print()


