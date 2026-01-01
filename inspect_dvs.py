import inspect
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
print(inspect.signature(CIFAR10DVS.__init__))
