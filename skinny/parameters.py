from .models import SkinnyBasic, SkinnyInception

from typing import Union

class Parameters:
    """
    Object holding training parameters.
    """
    def __init__(self, model: Union[SkinnyBasic, SkinnyInception], epochs: int, learning_rate: float, criterion, optimizer, device, batch_size: int):
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model.to(device=device)
        self.optimizer = optimizer(model.parameters(), lr=self.learning_rate, foreach=True)
