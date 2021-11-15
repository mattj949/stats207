import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)
PRINT_INTERVAL = 50


class CommodityLSTM(nn.Module):

    def __init__(self,
                validation_dataloader,
                 lr: float = 0.001,
                 hidden_size = 2,
                 num_layers = 1,
                 dropout = 0,
                 ) -> None:
        """LSTM constructor"""
        super(CommodityLSTM, self).__init__()

        self._lr = lr
        self.LSTM_l1 = nn.LSTM(input_size = 1,
                                hidden_size = hidden_size,
                                batch_first = True,
                                num_layers = num_layers,
                                dropout = dropout,
                                )
        self.linear = nn.Linear(hidden_size, 1)
        

        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)

        for _, val_batch in enumerate(validation_dataloader):
            # get validation loss 
            self.X_val, self.y_val = val_batch
            self.X_val = self.X_val.unsqueeze(-1).float()
            self.y_val = self.y_val.float()
            # TODO REMOVE validate on larger sample
            break

    def forward(self, x):
        """Makes a forward pass through the network."""

        nn_out, self.hidden = self.LSTM_l1(x)
        prediction = self.linear(nn_out[:,-1,:])
        return prediction

    def train(self, dataloader, validation_dataloader):
        """Trains the LSTM."""
        training_losses = []
        validation_losses = []

        for i_step, batch in enumerate(dataloader):
            self._optimizer.zero_grad()

            X, y = batch
            X = X.unsqueeze(-1).float()
            y = y.float()

            nn_output = self.forward(X)

            mseloss = nn.MSELoss()
            batch_loss = mseloss(nn_output, y)

            batch_loss.backward()
            self._optimizer.step()

            training_losses.append(batch_loss)

            val_batch_loss = 0
            nn_output_val = self.forward(self.X_val)
            mseloss = nn.MSELoss()
            val_batch_loss += mseloss(nn_output_val, self.y_val)
            validation_losses.append(val_batch_loss)
        
            if i_step % PRINT_INTERVAL == 0:
                print(f'TRAINING BATCH LOSS AT STEP {i_step}: {batch_loss}')
                print(f'VALIDATION BATCH LOSS AT STEP {i_step}: {val_batch_loss}')

        return training_losses, validation_losses

    def test(self, test_dataloader):
        predictions = []
        losses = []
        for i_step, batch in enumerate(test_dataloader):
            X, y = batch
            X = X.unsqueeze(-1).float()
            y = y.float()

            nn_output = self.forward(X)

            mseloss = nn.MSELoss()
            batch_loss = mseloss(nn_output, y)
            losses.append(batch_loss)
            predictions.append(nn_output)
        return predictions, losses
