import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

torch.manual_seed(1)
PRINT_INTERVAL = 700


class CommodityLSTM(nn.Module):

    def __init__(self,
                 #validation_dataloader,
                 lr: float = 0.0001,
                 hidden_size = 2,
                 num_layers = 1,
                 dropout = 0,
                 num_features = 1
                 ) -> None:
        """LSTM constructor"""
        super(CommodityLSTM, self).__init__()

        self._lr = lr
        torch.manual_seed(12)
        #self.bnorm1 = nn.BatchNorm1d(num_features)
        self.LSTM_l1 = nn.LSTM(input_size = num_features,
                                hidden_size = hidden_size,
                                batch_first = True,
                                num_layers = num_layers,
                                dropout = dropout,
                                )
        self.linear = nn.Linear(hidden_size, 1)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.sig_layer = nn.Sigmoid()
        

        # for _, val_batch in enumerate(validation_dataloader):
        #     # get validation loss 
        #     self.X_val, self.y_val = val_batch
        #     if self.num_features == 1:
        #         self.X_val = self.X_val.unsqueeze(-1).float()
        #     else:
        #         self.X_val = self.X_val.float()
        #     self.y_val = self.y_val.float()


    def forward(self, x):
        """Makes a forward pass through the network."""

        nn_out, (h_out, _) = self.LSTM_l1(x)
        prediction = self.linear(nn_out[:, -1, :])

        
        return prediction

    def train(self, dataloader, epochs = 2, validation_dataloader = None):
        """Trains the LSTM."""
        training_losses = []
        validation_losses = []
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        for epoch in tqdm(range(epochs)):
            for i_step, batch in enumerate(dataloader):
                self._optimizer.zero_grad()

                X, y = batch
                if self.num_features == 0:
                    X = X.unsqueeze(-1).float()
                else:
                    X = X.float()
                y = y.unsqueeze(-1).float()
            

                nn_output = self.forward(X)

                mseloss = nn.MSELoss()
                bceloss = nn.BCEWithLogitsLoss()
                #print(nn_output, y)
                
                batch_loss = mseloss(nn_output, y[:, 1, :])
                classification_loss = bceloss(nn_output, y[:, 0 , :])
                totalloss = batch_loss + classification_loss
                totalloss.backward()
                self._optimizer.step()

                batch_loss = totalloss
                training_losses.append(batch_loss)

                # val_batch_loss = 0
                # nn_output_val = self.forward(self.X_val)
                # mseloss = nn.MSELoss()
                # val_batch_loss += mseloss(nn_output_val, self.y_val)
                # validation_losses.append(val_batch_loss)
            
                if i_step % PRINT_INTERVAL == 0:
                    print(f'TRAINING BATCH LOSS AT STEP {i_step}: {batch_loss}')
                    #print(f'VALIDATION BATCH LOSS AT STEP {i_step}: {val_batch_loss}')

        return training_losses, validation_losses

    def test(self, test_dataloader):
        predictions = []
        actuals = []
        losses = []
        for i_step, batch in enumerate(test_dataloader):
            X, y = batch
            if self.num_features == 0:
                X = X.unsqueeze(-1).float()
            else:
                X = X.float()
            y = y.unsqueeze(-1).float()
            nn_output = self.forward(X)

            mseloss = nn.MSELoss()
            bceloss = nn.BCEWithLogitsLoss()
            #print(nn_output, y)
            
            batch_loss = mseloss(nn_output, y[:, 1, :])
            classification_loss = bceloss(nn_output, y[:, 0 , :])
            totalloss = batch_loss + classification_loss
            # nn_output = self.forward(X)

            # mseloss = nn.MSELoss()
            # batch_loss = mseloss(nn_output, y)
            # losses.append(batch_loss.detach().numpy())
            losses.append(totalloss.detach().numpy())

            predictions.append(nn_output)
            actuals.append(y)

        print('AVERAGE LOSS: ', np.mean(losses))
        predictions = np.concatenate([prediction.detach().numpy() for prediction in predictions])
        actuals = np.concatenate([actual[:, 1, :].detach().numpy() for actual in actuals]).flatten()
        
        return predictions, losses, actuals
