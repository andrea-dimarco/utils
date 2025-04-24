import math
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data


# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from imblearn.over_sampling import SMOTE
import utils



class FFNetwork(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 out_dim:int=1,
                 num_layers:int=1,
                 bias:bool=True,
                ) -> None:
        '''
        The Feed Forward Regressor model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `out_dim`: dimension of the output vector
            - `num_layers`: number of linear layers
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)

        self.data_dim = data_dim
        # Initialize Modules
        # input = ( batch_size, lookback*data_dim )
        model = [
            nn.Linear(in_features=data_dim, out_features=hidden_dim, bias=bias),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ]
        for _ in range(num_layers-1):
            model += [
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=bias),
                nn.ReLU(inplace=True),
            ]
        model += [
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ]
        self.feed = nn.Sequential(*model)
        # init weights
        self.feed.apply(init_weights)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, data_dim]

        Returns:
            - the predicted sequences [batch, data_dim]
        '''
        # x = (batch, data)
        x = self.feed(x)
        # x = ( batch, out_dim )
        return x



class FFClassifier(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 out_dim:int,
                 num_layers:int=1,
                 bias:bool=True,
                ) -> None:
        '''
        The Feed Forward Classifier model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `out_dim`: dimension of the output vector
            - `num_layers`: number of linear layers
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)

        self.data_dim = data_dim
        # Initialize Modules
        # input = ( batch_size, lookback*data_dim )
        model = [
            nn.Linear(in_features=data_dim, out_features=hidden_dim, bias=bias),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ]
        for _ in range(num_layers-1):
            model += [
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=bias),
                nn.ReLU(inplace=True),
            ]
        model += [
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
        ]
        self.softmax = nn.Softmax(dim=1) # dim means over which dimension the softmax is to be performed
        self.feed = nn.Sequential(*model)
        # init weights
        self.feed.apply(init_weights)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, data_dim]

        Returns:
            - the predicted sequences [batch, data_dim]
        '''
        # x = (batch, data)
        x = self.feed(x)
        # x = ( batch, out_dim )
        x = self.softmax(x)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # (max_len, 1, embedding_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class CTransformer(nn.Module):

    def __init__(self,
                 embedding_size:int,
                 n_head:int,
                 n_encoder_layers:int,
                 n_decoder_layers:int,
                 dim_feedforward:int,
                 device,
                 dropout:float=0.1,
                 name:str="CTransformer"
                ):
        super().__init__()
        self.__name = name
        self.__transformer = nn.Transformer(d_model=embedding_size,
                                            nhead=n_head,
                                            num_encoder_layers=n_encoder_layers,
                                            num_decoder_layers=n_decoder_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                        ).to(device)
        self.__positional_encoder = PositionalEncoding(embedding_size=embedding_size)

        # TODO: this

    def name(self) -> str:
        return self.__name
        
    def forward(x:torch.Tensor) -> torch.Tensor:
        return x




def build_auc_plot(classifiers, X_data, y_data, file_path:str):
    fig, ax = plt.subplots()

    for classifier in classifiers:
        model_name = type(classifier).__name__
        y_pred = classifier.predict(X_data)
        y_pred = y_pred.argmax(axis=-1)
        fpr, tpr, _ = metrics.roc_curve(y_data,  y_pred)
        auc2 =  roc_auc_score(y_data, y_pred)
        plt.plot(fpr,tpr,label=f"{model_name}, auc={int(round(auc2*100))}%")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.savefig(file_path, dpi=200)
    plt.clf()



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def set_seed(seed=0) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False



def get_transformer(embedding_size:int,
                    n_head:int,
                    n_encoder_layers:int,
                    n_decoder_layers:int,
                    dim_feedforward:int,
                    device,
                    dropout:float=0.1
                    ) -> nn.Transformer:
    '''
    Get an untrained Transformer model
    '''
    transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=n_head,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)
    return transformer



def train_model(model,
                X_train:torch.Tensor,
                y_train:torch.Tensor,
                X_val:torch.Tensor,
                y_val:torch.Tensor,
                n_epochs:int,
                batch_size:int,
                device,
                adam_lr:float,
                adam_b1:float,
                adam_b2:float,
                decay_start:float=1.0,
                decay_end:float=1.0,
                plot_loss:bool=False,
                loss_plot_folder:str="/data/results/img/",
                model_name:str="DeepModel",
                loss_fn=nn.CrossEntropyLoss(),
                val_frequency:int=100,
                save_folder:str="/data/models/",
                verbose:bool=True,
               ):
    '''
    Instanciate, train and return the model the model.

    Arguments:
        - `X_train`: train Tensor [n_sequences, data_dim]
        - `y_train`: the targets
        - `X_val`: seques for validation
        - `y_val`: targets for validation
        - `plot_loss`: if to plot the loss or not
        - `loss_fn`: the loss function to use
        - `val_frequency`: after how many epochs to run a validation epoch
    '''
    input_size:int = X_train.size()[1]
    if verbose:
        print(f"Using device {device}.")

    if verbose:
        print("Parameters count: ", count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=adam_lr, betas=(adam_b1, adam_b2))
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                               start_factor=decay_start,
                                               end_factor=decay_end,
                                               total_iters=n_epochs
                                              )
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                                   shuffle=True,
                                   batch_size=batch_size
                                  )

    if verbose:
        print("Training begins.")
    loss_history = []
    if verbose:
        timer = utils.TimeExecution()
        timer.start()
    for epoch in range(n_epochs):
        # Training step
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch.to(device=device))
            loss = loss_fn(y_pred, y_batch.to(device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step every val_frequency epochs
        if (epoch % val_frequency) == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_train)
                train_loss = loss_fn(y_pred, y_train)
                y_pred = model(X_val)
                val_loss = loss_fn(y_pred, y_val)
                if plot_loss:
                    loss_history.append(val_loss.item())
            if verbose:
                timer.end()
                print("Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, lr=%.4f, elapsed_time=%.2fs" % (epoch, n_epochs, train_loss, val_loss, optimizer.param_groups[0]["lr"], timer.elapsed()))
                timer.start()
        lr_scheduler.step()
    
    # Save loss plot
    if plot_loss:
        plt.plot(loss_history, label="validation loss")
        plt.savefig(f"{loss_plot_folder}{model_name}-loss.png")

    # Log the trained model
    if not (save_folder is None):
        torch.save(model.state_dict(), f"{save_folder}{model_name}.pth")
    return model




class REPORT:

    @classmethod
    def print_report(name, report):
        cv = len(list(report.values())[0]['cross_val'])

        rows = [
                [ '-', 'precision', 'recall', 'f1-score', 'support', 'roc', 'confusion matrix', f'cross val (cv={cv})' ],
                [ '(class)', ['0', '1'], ['0', '1'], ['0', '1'], ['0', '1'], '-', '-', '-' ],
               ]

        for model_name, data in report.items():
            metrics = [ [ data['metrics'][c][metric] for c in [ '0', '1' ] ] for metric in [ 'precision', 'recall', 'f1-score', 'support' ] ]
            row = [model_name, *metrics, data['roc'], REPORT.render_confusion_matrix(data['confusion_matrix']), REPORT.render_cross_val(data['cross_val'])]
            rows.append(row)

        print(REPORT.build_table(rows, heading=name))

    @classmethod
    def build_table(rows, heading=None):
        if not rows:
            return ""

        rows = list(map(lambda x: REPORT.build_row(x), rows))
        row_separator = "-" * len(rows[0])
        output = row_separator + "\n"

        if heading:
            output += REPORT.build_row([ heading ], width=len(rows[0]) - 2) + "\n" + row_separator + "\n"

        for row in rows:
            output += row + "\n" + row_separator + "\n"

        return output

    @classmethod
    def render_cross_val(cv):
        return f"{cv.mean()*100:.2f}% ({cv.std():.2f})"
    
    @classmethod
    def render_confusion_matrix(cf):
        return f"{cf[0][0]} {cf[0][1]} - {cf[1][0]} {cf[1][1]}"
    
    @classmethod
    def build_row(components, width=31):
        center = "|".join([REPORT.build_cell(component, width=width) for component in components])
        row = f"|{center}|"
        return row
    
    @classmethod
    def build_cell(component, width=31):
        if type(component) == str:
            return component.center(width)
        elif type(component) == int:
            return f"{component}".center(width)
        elif type(component) == np.float64 or type(component) == float:
            return f"{component*100:.2f}%".center(width)
        elif type(component) == list or type(component) == set:
            return "|".join(map(lambda x: REPORT.build_cell(x, width=width//len(component)), component))
        else:
            print(component)
            print(type(component))

    @classmethod
    def create_report(X_data, y_data, classifiers, cv=5):
        report = {}

        for model_name, classifier in classifiers:
            y_pred = classifier.predict(X_data)
            y_pred = y_pred.argmax(axis=-1)

            # estimator = KerasClassifier(build_fn= lambda : classifier, batch_size = 1, epochs = 100)
            report[model_name] = {}
            report[model_name]['roc'] = roc_auc_score(y_data, y_pred)
            report[model_name]['confusion_matrix'] = confusion_matrix(y_data, y_pred)
            #cambiamento di X, Y in features, label
            report[model_name]['cross_val'] = np.array([]) #cross_val_score(estimator, features, label, cv=cv)
            report[model_name]['metrics'] = REPORT.classification_report(y_data, y_pred, output_dict=True)

        return report
    
    @classmethod
    def show_details(X_data, y_data, model, cv):
        model_name, classifier = model
        y_pred = classifier.predict(X_data)
        y_pred = y_pred.argmax(axis=-1)
        #cambiamento di X, Y in features, label
        # estimator = KerasClassifier(build_fn= lambda : classifier, batch_size = 1, epochs = 100)
        # scores = cross_val_score(estimator, features, label, cv=cv)
        class_report = REPORT.classification_report(y_data, y_pred)
        confusion = confusion_matrix(y_data, y_pred)
        roc = roc_auc_score(y_data, y_pred)
        _, accuracy = classifier.evaluate(X_data, y_data, verbose=1)

        print(f" {model_name} ".center(60, "="))

        print(class_report)
        print(confusion)
        print(f"{model_name} roc: {roc*100:.2f}%")
        # print(scores)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(f"{accuracy * 100}% accuracy")

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["0", "1"])
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(model_name)
        plt.show()

        build_auc_plot([classifier], X_data, y_data)

        print("\n\n")


