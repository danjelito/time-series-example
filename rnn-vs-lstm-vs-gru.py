import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sns.set_style("whitegrid")

debug = False


def clean_df(df):
    return (
        df.loc[:, ["date", "meantemp"]]
        .sort_values("date", ascending=True)
        .assign(
            date=lambda df_: pd.to_datetime(df_["date"]),
            meantemp=lambda df_: df_["meantemp"].astype("float32"),
        )
        .set_index("date")
        .resample("1d")
        .ffill()
        .reset_index()
    )


def plot_predictions(
    all_y,
    rnn_preds,
    lstm_preds,
    gru_preds,
    dummy_preds,
    rnn_loss,
    lstm_loss,
    gru_loss,
    dummy_loss,
):
    x_axis = range(len(all_y))

    plt.figure(figsize=(4, 8))
    plt.plot(x_axis, all_y, label="True", linestyle="-", color="black")
    plt.plot(
        x_axis,
        dummy_preds,
        label=f"Dummy prediction, loss {dummy_loss: .4f}",
        linestyle="--",
        color="red",
    )
    plt.plot(
        x_axis,
        rnn_preds,
        label=f"RNN prediction, loss {rnn_loss: .4f}",
        linestyle="--",
        color="blue",
    )
    plt.plot(
        x_axis,
        lstm_preds,
        label=f"LSTM prediction, loss {lstm_loss: .4f}",
        linestyle="--",
        color="green",
    )
    plt.plot(
        x_axis,
        gru_preds,
        label=f"GRU prediction, loss {gru_loss: .4f}",
        linestyle="--",
        color="yellow",
    )

    plt.title("Prediction Result")
    plt.legend()
    plt.show()


def create_timeseries_dataset(dataset, look_back=1):
    """
    Creates a dataset for time series prediction with a specified look_back window.

    Args:
        dataset: The original time series data as a NumPy array.
        look_back: The number of previous time steps to use as input for prediction.

    Returns:
        A tuple containing:
            - xs: A NumPy array containing the input sequences.
            - ys: A NumPy array containing the target values.
    """
    xs, ys = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back)]
        xs.append(a)
        ys.append(dataset[i + look_back])
    return np.array(xs), np.array(ys)


# --- Data preparation ---


# Read DF
print("Reading DF...")
train_path = "input/delhi-climate/DailyDelhiClimateTrain.csv"
test_path = "input/delhi-climate/DailyDelhiClimateTest.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Clean DF
print("Cleaning DF...")
df_train = clean_df(df_train)
df_test = clean_df(df_test)
print("    DF shape: ", df_train.shape, df_test.shape)

# Create dataset
print("Creating dataset...")
x_train = df_train["meantemp"].values[:-114]
x_val = df_train["meantemp"].values[-114:]
x_test = df_test["meantemp"].values
x_train_dataset, y_train = create_timeseries_dataset(x_train, 12)
x_val_dataset, y_val = create_timeseries_dataset(x_val, 12)
x_test_dataset, y_test = create_timeseries_dataset(x_test, 12)
assert all(
    [x_train_dataset.ndim == 2, x_val_dataset.ndim == 2, x_test_dataset.ndim == 2]
)
assert all([y_train.ndim == 1, y_val.ndim == 1, y_test.ndim == 1])


# Wrap out dataset in Pythorch's dataset and dataloader
class TimeseriesDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index):
        # x should be [batch_size, seq_len, features]
        x = torch.from_numpy(np.expand_dims(self.x[index], 1)).float()
        # y should be [batch_size, 1]
        y = torch.from_numpy(np.expand_dims(self.y[index], 0)).float()
        return x, y


batch_size = 8
train_dataset = TimeseriesDataset(x_train_dataset, y_train)
val_dataset = TimeseriesDataset(x_val_dataset, y_val)
test_dataset = TimeseriesDataset(x_test_dataset, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check if dataset is correctly implemented
if debug:
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"    Batch {batch_idx + 1}")
        print(
            f"    Inputs shape: {inputs.shape}"
        )  # Should be # x should be [batch_size, seq_len, features]
        print(f"    Targets shape: {targets.shape}")  # Should be (batch_size)
        print("    Inputs:", inputs)
        print("    Targets:", targets)
        break


# --- Model preparation ---


# Create model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:, -1, :]  # Last step output
        output = self.linear(output)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # Last step output
        output = self.linear(output)
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        output = output[:, -1, :]  # Last step output
        output = self.linear(output)
        return output


# --- Training setup ---


# Initialize the RNN model and related components
rnn = RNN(
    input_size=1,
    hidden_size=128,
    output_size=1,
    num_layers=2,
)
rnn_criterion = nn.MSELoss()
rnn_optimizer = optim.Adam(rnn.parameters(), lr=2e-5)
rnn_scheduler = ReduceLROnPlateau(rnn_optimizer, patience=1, factor=0.5)

# Initialize the LSTM model and related components
lstm = LSTM(
    input_size=1,
    hidden_size=128,
    output_size=1,
    num_layers=2,
)
lstm_criterion = nn.MSELoss()
lstm_optimizer = optim.Adam(lstm.parameters(), lr=5e-5)
lstm_scheduler = ReduceLROnPlateau(lstm_optimizer, patience=1, factor=0.5)

# Initialize the GRU model and related components
gru = GRU(
    input_size=1,
    hidden_size=128,
    output_size=1,
    num_layers=2,
)
gru_criterion = nn.MSELoss()
gru_optimizer = optim.Adam(gru.parameters(), lr=5e-5)
gru_scheduler = ReduceLROnPlateau(gru_optimizer, patience=1, factor=0.5)


# --- Training, validaton and test function ---


def train_one_step(model, optimizer, loss_fn, x, y):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0.0
    for batch_idx, (x, y) in enumerate(
        tqdm(dataloader, leave=False, desc="    Training")
    ):
        loss = train_one_step(model, optimizer, loss_fn, x, y)
        total_loss += loss
    return total_loss / (batch_idx + 1)


def val_one_step(model, optimizer, loss_fn, x, y):
    output = model(x)
    loss = loss_fn(output, y)
    return loss.item()


def val_one_epoch(model, optimizer, loss_fn, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(
            tqdm(dataloader, leave=False, desc="    Validation")
        ):
            loss = val_one_step(model, optimizer, loss_fn, x, y)
            total_loss += loss
    return total_loss / (batch_idx + 1)


def test_one_step(model, loss_fn, x, y):
    """
    Performs one testing step.
    """
    output = model(x)
    loss = loss_fn(output, y)
    # preds = torch.argmax(output, dim=1)
    return loss.item(), x, y, output


def test_one_epoch(model, loss_fn, dataloader):
    """
    Performs testing for one epoch over the given dataset.
    """
    model.eval()
    total_loss = 0.0
    all_x, all_y, all_preds = [], [], []
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(
            tqdm(dataloader, desc="    Testing", leave=False)
        ):
            # Perform one testing step
            loss, step_x, step_y, step_preds = test_one_step(
                model, loss_fn, batch_x, batch_y
            )
            total_loss += loss
            # Accumulate results
            all_x.append(step_x)
            all_y.append(step_y)
            all_preds.append(step_preds)

    # Concatenate results for final output
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    # Return average loss and all results
    return total_loss / len(dataloader), all_x, all_y, all_preds


def train_and_validate(
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    print_step,
    patience,
):
    train_losses = []
    val_losses = []
    last_loss = float("inf")  # Initialize last_loss to infinity
    counter = 0  # Initialize early stopping counter

    for epoch in range(epochs):
        # Training
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss = val_one_epoch(model, optimizer, criterion, val_loader)
        val_losses.append(val_loss)

        # Print score
        if (epoch == 0) or ((epoch + 1) % print_step == 0):
            avg_train_loss = np.mean(train_loss)
            avg_val_loss = np.mean(val_loss)
            print(
                f"    Epoch {epoch+1: <3}/{epochs} | train loss = {avg_train_loss: .8f} | val loss = {avg_val_loss: .8f}"
            )

        # Lr scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < last_loss:
            last_loss = val_loss
            counter = 0
        else:
            counter += 1
            last_loss = val_loss
            if counter >= patience:
                print("    Early stopping triggered")
                break

    return train_losses, val_losses


# --- Training argument ---


if debug:
    epochs = 1
else:
    epochs = 100

print_step = 1
train_losses = []
val_losses = []
patience = 3  # Number of epochs to wait for improvement


# --- Training loop ---


print("Training...")
rnn_train_losses, rnn_val_losses = train_and_validate(
    model=rnn,
    optimizer=rnn_optimizer,
    criterion=rnn_criterion,
    scheduler=rnn_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    print_step=print_step,
    patience=patience,
)

lstm_train_losses, lstm_val_losses = train_and_validate(
    model=lstm,
    optimizer=lstm_optimizer,
    criterion=lstm_criterion,
    scheduler=lstm_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    print_step=print_step,
    patience=patience,
)
gru_train_losses, gru_val_losses = train_and_validate(
    model=gru,
    optimizer=gru_optimizer,
    criterion=gru_criterion,
    scheduler=gru_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    print_step=print_step,
    patience=patience,
)


# --- Testing ---


print("Testing...")

# RNN
rnn_loss, all_x, all_y, rnn_preds = test_one_epoch(
    rnn, loss_fn=rnn_criterion, dataloader=test_loader
)

# LSTM
lstm_loss, _, _, lstm_preds = test_one_epoch(
    lstm, loss_fn=lstm_criterion, dataloader=test_loader
)
# GRU
gru_loss, _, _, gru_preds = test_one_epoch(
    gru, loss_fn=gru_criterion, dataloader=test_loader
)

# Last timestep prediction
dummy_preds = all_x[:, -1, 0].unsqueeze(-1)
dummy_loss = rnn_criterion(all_y, dummy_preds).item()


# --- Plotting ---


plot_predictions(
    all_y,
    rnn_preds,
    lstm_preds,
    gru_preds,
    dummy_preds,
    rnn_loss,
    lstm_loss,
    gru_loss,
    dummy_loss,
)
