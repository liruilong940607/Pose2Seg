class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeters():
    def __init__(self):
        self.meters = {}

    def __len__(self):
        return len(self.meters)

    def __getitem__(self, key):
        if key not in self.meters:
            self.meters[key] = AverageMeter()
        return self.meters[key]
    
    def clear(self):
        self.meters = {}
    
    def items(self):
        return self.meters.items()

        
