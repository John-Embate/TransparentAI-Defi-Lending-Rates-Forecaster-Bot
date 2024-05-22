#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import os

from dotenv import load_dotenv
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

#For Data Handling and DL Modeling
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

#For GizaTech Libraries and On chain proof of inference
from giza_actions.model import GizaModel
from giza_actions.action import Action, action
from giza_actions.task import task


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


single_test_sample = None

@task(name='Generate Prediction')
def predict_output(data_for_prediction, model_id, version_id):
    # Initialize the prediction model with the given model and version IDs.
    prediction_model = GizaModel(id=model_id, version=version_id)
    
    # Perform prediction using the model's predict method and ensure output integrity.
    prediction_result, prediction_tracking_id = prediction_model.predict(
        input_feed={"model_input": data_for_prediction}, 
        verifiable=True,
        job_size="XL",
    )
    print(prediction_result)
    print(prediction_tracking_id)
    return prediction_result, prediction_tracking_id

@action(name='Execute Prediction Process Supply', log_prints=True)
def execute_prediction_supply():
    model_identifier = 368  # Customize with the specific model ID for Supply Rate Model
    version_identifier = 11  # Customize with the specific version ID for Supply Rate Model
    
    # Using a single sample from the test set for making a prediction
    prediction_result, operation_tracking_id = predict_output(single_test_sample, model_identifier, version_identifier) 
    return prediction_result, operation_tracking_id

@action(name='Execute Prediction Process Borrow', log_prints=True)
def execute_prediction_borrow():
    model_identifier = 619  # Customize with the specific model ID for Supply Rate Model
    version_identifier = 1  # Customize with the specific version ID for Supply Rate Model
    
    # Using a single sample from the test set for making a prediction
    prediction_result, operation_tracking_id = predict_output(single_test_sample, model_identifier, version_identifier) 
    return prediction_result, operation_tracking_id

def main() -> None:

    # Load the CSV files for smaple predictions
    X_df = pd.read_csv('X_tensor_Compound_V2.csv')
    y_df = pd.read_csv('y_tensor_Compound_V2.csv')
    
    # Convert DataFrames to tensors
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.values.squeeze(), dtype=torch.float32)  # Assuming y is 1D
    
    
    """Start the bot."""
    load_dotenv()

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.environ["TOKEN"]).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
