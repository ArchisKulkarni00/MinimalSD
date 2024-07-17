import datetime
import os

import diffusers
import yaml
import time
import logging

logger = None


def load_yaml_file(filename):
    config = None
    try:
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)
    except:
        print("There was an error reading the yaml file.")

    return config


def process_presets(preset_name, preset_file='presets.yml'):
    presets_file = load_yaml_file(preset_file)
    return presets_file[preset_name][0]['positive_prompt'], presets_file[preset_name][1]['negative_prompt']


def generate_unique_filename(prefix, index):
    timestamp = time.strftime('%m%d_%H%M%S')
    return f'{prefix}_{timestamp}_{index + 1}.png'


def print_main_menu():
    print("=== Image Generator Menu ===")
    print("[1] Load Model    [2] Generate Images    [3] Remove Model")
    print("[4] Display GPU details    [5] Exit")
    print("==========================")


def initialize_logging():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    os.makedirs("logs", exist_ok=True)

    # Formatter for log messages
    formatter_others = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s', '%H:%M:%S')
    formatter_info = logging.Formatter('%(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the level as needed
    console_handler.setFormatter(formatter_info)
    logger.addHandler(console_handler)

    # File handler - logs based on date
    today_date = datetime.date.today().strftime('%Y-%m-%d')
    log_file = os.path.join("logs", f"log_{today_date}.log")
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Set the level as needed
    file_handler.setFormatter(formatter_others)
    logger.addHandler(file_handler)

    logger.debug("Initialized debugger. Session started at {}".format(time.strftime('%d-%m-%Y | %H:%M:%S')))
    return logger
