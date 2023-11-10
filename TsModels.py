import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import tensorflow as tf
import sklearn as sk




#@title Define the functions that build and train a model
def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    # A sequential model contains one or more layers.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    # model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.add(tf.keras.layers.Dense(128, activation="relu", input_shape=(1,)))
    # input_dim=6
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="relu"))


# Compile the model topography into code that
    # TensorFlow can efficiently execute. Configure
    # training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, feature, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the feature values and the label values to the
    # model. The model will train for the specified number
    # of epochs, gradually learning how the feature values
    # relate to the label values.
    history = model.fit(x=feature,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of history.
    epochs = history.epoch

    # Gather the history (a snapshot) of each epoch.
    hist = pd.DataFrame(history.history)

    # Specifically gather the model's root mean
    # squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

print("Defined build_model and train_model")


#@title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against the training feature and label."""
    # create new figure such that it does not mess with other figures
    plt.figure()
    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    # x0 = 0
    y0 = trained_bias
    # x1 = feature[-1]
    # y1 = trained_bias + (trained_weight * x1)
    print(f'len bias: {len(y0)}, bias = {y0}')
    print(f'len weight: {len(trained_weight[0])}, trained weight = {trained_weight}')
    plt.plot(np.array(feature), trained_bias + np.transpose(np.array(feature)*trained_weight))

    # Render the scatter plot and the red line.

def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])

print("Defined the plot_the_model and plot_the_loss_curve functions.")

def predict_values(n, feature: str, label, test_data: pd.DataFrame, model):
    """Predict house values based on a feature. This is a pretty specific function so not sure if its useful for other situations."""

    batch = test_data[feature][10000:10000 + n]
    predicted_values = model.predict_on_batch(x=batch)

    print("feature   label          predicted")
    print("  value   value          value")
    print("          in thousand$   in thousand$")
    print("--------------------------------------")
    for i in range(n):
        print ("%5.0f %6.0f %15.0f" % (test_data[feature][10000 + i],
                                       test_data[label][10000 + i],
                                       predicted_values[i][0] ))