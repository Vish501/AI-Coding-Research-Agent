import os
import yaml
import json
import joblib
import base64

from CodeResearcher.utils.logger import get_logger
from box.exceptions import BoxValueError
from typeguard import typechecked as ensure_annotations          # Enforces runtime validation of the function’s argument and return types
from box import ConfigBox
from pathlib import Path
from typing import Any, Union

logger = get_logger()

"""
### Functions in the Script
- read_yaml(path: Path) -> ConfigBox
- create_directories(path_list: list, verbose: bool=True) -> None
- load_json(path: Path) -> ConfigBox
- save_bin(save_path: Path, data: Any) -> None
- load_bin(path: Path) -> Any
- get_kb_size(path: Path) -> str
- decode_image_Base64(image_string: str, save_path: Union[str, Path]) -> None
- encode_image_Base64(path: Union[str, Path]) -> str
- convert_paths_to_str(obj: dict) -> dict
"""


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """
    Reads a YAML file from the given path and returns a ConfigBox object (like a dictionary with dot access).

    Parameters:
    - path (Path): Path to the YAML file.

    Returns:
    - ConfigBox: Parsed YAML content as a dot-accessible dictionary.

    Raises:
    - FileNotFoundError: If the YAML file does not exist.
    - BoxValueError: If the YAML file is malformed or empty.
    - yaml.YAMLError: If YAML syntax is invalid.
    - Exception: For any other unhandled exceptions.
    """
    try:
        # Check if the file exists
        if not path.exists():
            logger.error(f"YAML file not found at: {path}")
            raise FileNotFoundError(f"YAML file not found at: {path}")
        
        # Check if the file has a valid YAML extension
        if path.suffix.lower() not in [".yaml", ".yml"]:
            logger.error(f"Invalid file type: {path.name} is not a YAML file.")
            raise FileNotFoundError(f"Invalid file type: {path.name} is not a YAML file.")
        
        # Attempt to read and parse the YAML file
        with open(path, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path} loaded successfully")
            return ConfigBox(content)

    except FileNotFoundError as exception_error:
        # Log and re-raise file not found errors
        logger.error(f"File not found: {exception_error}")
        raise exception_error

    except BoxValueError:
        # Raised if ConfigBox receives invalid (e.g., None) content
        logger.error(f"Failed to parse YAML into ConfigBox: Possibly empty or malformed.")
        raise ValueError("YAML file is either empty or invalid")
    
    except yaml.YAMLError as exception_error:
        logger.error(f"Failed to parse YAML syntax: {exception_error}")
        raise ValueError(f"Invalid YAML format in file: {path}")
    
    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while reading YAML file: {exception_error}")
        raise exception_error


@ensure_annotations
def create_directories(path_list: list, verbose: bool=True) -> None:
    """
    Creates directories from the given list of paths if they don't already exist.

    Parameters:
    - path_list (list[Union[str, Path]]): List of directory paths to create.
    - verbose (bool): If True, prints log messages when directories are created.

    Raises:
    - TypeError: If path_list contains invalid types.
    - OSError: If directory creation fails due to permission or filesystem error.
    """
    for path in path_list:
        try:
            if not isinstance(path, (str, Path)):
                logger.error(f"Invalid path: {path} must be a str or Path.")
                raise TypeError(f"Invalid path: {path} must be a str or Path.")
            
            # Attempt to create the directory
            path_object = Path(path)
            
            path_object.mkdir(parents=True, exist_ok=True)
            if verbose:
                logger.info(f"Directory: {path_object} created successfully.")
        
        except OSError as exception_error:
            logger.error(f"OS error while creating {path}: {exception_error}")
            raise exception_error

        except Exception as exception_error:
            # Catch-all for other exceptions
            logger.error(f"Unexpected error while creating {path}: {exception_error}")
            raise exception_error


@ensure_annotations
def save_json(save_path: Path, data: Any) -> None:
    """
    Saves the provided data as a JSON file to the specified path.

    Parameters:
    - save_path (Path): Destination path for the JSON file.
    - data (Any): Serializable data (usually dict or list) to store in JSON format.

    Raises:
    - TypeError: If data contains unserializable objects.
    - FileNotFoundError: If the directory path does not exist.
    - PermissionError: If the program lacks permission to write.
    - Exception: For any other unhandled errors.
    """
    try:
        # Ensure parent directory exists using our utility function
        create_directories([save_path.parent])

        # Write JSON to file
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        logger.info(f"JSON file saved at: {save_path}")

    except TypeError as exception_error:
        # Raised when the data can't be serialized (e.g., a set, custom object, or a NumPy array without conversion).
        logger.error(f"Failed to seialize data to JSON at {save_path}: {exception_error}")
        raise exception_error

    except FileNotFoundError as exception_error:
        # Raised if the path is invalid (rare here because we’re creating it, but still a good fallback).
        logger.error(f"Directory not found for saving JSON at: {save_path.parent}")
        raise exception_error

    except PermissionError as exception_error:
        # Happens when we try saving to protected locations
        logger.error(f"Permission denied to save file at: {save_path}")
        raise exception_error

    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while saving JSON to {save_path}: {exception_error}")
        raise exception_error


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Parameters:
    - path (Path): Path to the JSON file.

    Returns:
    - ConfigBox: Parsed JSON content with dot-access support.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - BoxValueError: If the JSON is malformed or empty.
    - PermissionError: If the file cannot be read due to permission issues.
    - json.JSONDecodeError: If the file isn't valid JSON.
    - Exception: For other unexpected issues.
    """
    try:
        if not path.exists():
            logger.error(f"JSON file not found at: {path}")
            raise FileNotFoundError(f"JSON file not found at: {path}")
        
        # Check if the file has a valid YAML extension
        if path.suffix.lower() not in [".json"]:
            logger.error(f"Invalid file type: {path.name} is not a JSON file.")
            raise ValueError(f"Invalid file type: {path.name} is not a JSON file.")

        # Attempt to read a load JSON file
        with open(path, "r", encoding="utf-8") as file:
            content = json.load(file)
            logger.info(f"JSON file succesfully loaded form: {path}")
            return ConfigBox(content)

    except FileNotFoundError as exception_error:
        # Log and re-raise file not found errors
        logger.error(f"File not found: {exception_error}")
        raise exception_error

    except BoxValueError as exception_error:
        # Raised if ConfigBox receives invalid (e.g., None) content
        logger.error(f"Failed to parse JSON into ConfigBox: Possibly empty or malformed: {exception_error}")
        raise ValueError("JSON file is either empty or invalid")
    
    except PermissionError as exception_error:
        logger.error(f"Permission denied while reading JSON file at: {path}")
        raise exception_error

    except json.JSONDecodeError as exception_error:
        # Handles JSON syntax issues explicitly (e.g. missing commas, braces).
        logger.error(f"Failed to decode JSON file at: {path}")
        raise exception_error
    
    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while reading JSON file: {exception_error}")
        raise exception_error


@ensure_annotations
def save_bin(save_path: Path, data: Any) -> None:
    """
    Serializes and saves data to a binary (.bin or .pkl) file using joblib.

    Parameters:
    - save_path (Path): File path where the binary file will be saved.
    - data (Any): Any Python object to be serialized and stored.

    Raises:
    - FileNotFoundError: If the directory to save the file doesn't exist.
    - PermissionError: If write access is denied at the save path.
    - TypeError: If the object cannot be serialized by joblib.
    - Exception: For any other unexpected errors.
    """
    try:
        # Ensure parent directory exists using our utility function
        save_path = Path(save_path)
        create_directories([save_path.parent])

        # Serialize and save the data using joblib
        joblib.dump(value=data, filename=save_path)
        logger.info(f"Binary file saved at: {save_path}")

    except FileNotFoundError as exception_error:
        # Raised if the path is invalid (rare here because we’re creating it, but still a good fallback).
        logger.error(f"Directory not found for saving binary file at: {save_path.parent}")
        raise exception_error

    except PermissionError as exception_error:
        # Happens when we try saving to protected locations
        logger.error(f"Permission denied to save file at: {save_path}")
        raise exception_error
    
    except TypeError as exception_error:
        # Raised when the data can't be serialized
        logger.error(f"Failed to seialize data to a binary file at {save_path}: {exception_error}")
        raise exception_error

    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while saving JSON to {save_path}: {exception_error}")
        raise exception_error


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads and deserializes a binary file.

    Parameters:
    - path (Path): Path to the binary file.

    Returns:
    - Any: The original Python object that was serialized.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - PermissionError: If the file cannot be read due to permission issues.
    - ValueError: If the file has an incorrect extension.
    - Exception: For other unexpected issues.
    """
    try:
        if not path.exists():
            logger.error(f"Binary file not found at: {path}")
            raise FileNotFoundError(f"Binary file not found at: {path}")
        
        # Check if the file has a valid YAML extension
        if path.suffix.lower() not in [".bin", ".pkl"]:
            logger.error(f"Invalid file type: {path.name} is not a .bin or .pkl file.")
            raise ValueError(f"Invalid file type: {path.name} is not a .bin or .pkl file.")
        
        with open(path, "rb") as file:
            content = joblib.load(file)
            logger.info(f"Binary file succesfully loaded form: {path}")
            return content
    
    except FileNotFoundError as exception_error:
        # Log and re-raise file not found errors
        logger.error(f"File not found: {exception_error}")
        raise exception_error
    
    except PermissionError as exception_error:
        logger.error(f"Permission denied while reading Binary file at: {path}")
        raise exception_error
    
    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while reading Binary file: {exception_error}")
        raise exception_error


@ensure_annotations
def get_kb_size(path: Path) -> str:
    """
    Returns the size of a file in kilobytes (KB).

    Parameters:
    - path (Path): Path to the file whose size is to be computed.

    Returns:
    - str: File size in kilobytes, as a string with 'KB' suffix.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - IsADirectoryError: If the path points to a directory, not a file.
    - PermissionError: If the file exists but access is denied.
    - Exception: For any other unexpected errors.
    """
    try:
        if not path.exists():
            logger.error(f"File not found at: {path}")
            raise FileNotFoundError(f"File not found at: {path}")

        if path.is_dir():
            logger.error(f"Expected a file but got a directory: {path}")
            raise IsADirectoryError(f"{path} is a directory, not a file.")
        
        size_in_kb = round(os.path.getsize(path) / 1024)
        return f"~ {size_in_kb} KB"

    except FileNotFoundError as exception_error:
        logger.error(f"{exception_error}")
        raise exception_error

    except PermissionError as exception_error:
        logger.error(f"Permission denied while accessing file at: {path}")
        raise exception_error

    except IsADirectoryError as exception_error:
        logger.error(f"{exception_error}")
        raise exception_error
    
    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while accessing file at {path}: {exception_error}")
        raise exception_error


@ensure_annotations
def decode_image_Base64(image_string: str, save_path: Union[str, Path]) -> None:
    """
    Decodes a base64 string and writes it as an image file.

    Parameters:
    - image_string (str): Base64-encoded image string.
    - save_path (str or Path): File path where the image should be saved.

    Raises:
    - FileNotFoundError: If the directory path does not exist.
    - PermissionError: If the program lacks permission to write.
    - ValueError: If the base64 string is malformed or empty.
    - Exception: For any other unexpected errors. 
    """
    try:
        # Ensure parent directory exists using our utility function
        save_path = Path(save_path)
        create_directories([save_path.parent])

        # Making sure the image string provided is not empty
        if not image_string.strip():
            logger.error("Provided base64 string is empty.")
            raise ValueError("Base64 string cannot be empty.")

        # Process string to image
        image_bytes = base64.b64decode(image_string)
        with open(save_path, "wb") as file:
            file.write(image_bytes)
            logger.info(f"Image decoded and saved at: {save_path}")

    except FileNotFoundError as exception_error:
        # Raised if the path is invalid (rare here because we’re creating it, but still a good fallback).
        logger.error(f"Directory not found for saving image at: {save_path.parent}")
        raise exception_error

    except PermissionError as exception_error:
        # Happens when we try saving to protected locations
        logger.error(f"Permission denied to save image at: {save_path}")
        raise exception_error

    except base64.binascii.Error as exception_error:
        logger.error(f"Failed to decode base64 string: {exception_error}")
        raise ValueError("Invalid base64-encoded data.")

    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while saving image to {save_path}: {exception_error}")
        raise exception_error


@ensure_annotations
def encode_image_Base64(path: Union[str, Path]) -> str:
    """
    Reads an image file and returns its base64-encoded content as a UTF-8 string.

    Parameters:
    - image_path (str or Path): Path to the image file.

    Returns:
    - str: Base64-encoded image content as a string.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - PermissionError: If the file cannot be accessed due to permission.
    - IsADirectoryError: If the path is a directory.
    - Exception: For any other unexpected errors.
    """
    try:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise IsADirectoryError(f"Object found is not a file at: {path}")

        with open(path, "rb") as file:
            image_string = base64.b64encode(file.read())
            encoded_string = image_string.decode("utf-8")
            logger.info(f"Image successfully encoded to base64 from: {path}")
            return encoded_string

    except FileNotFoundError as exception_error:
        logger.error(f"{exception_error}")
        raise exception_error

    except PermissionError as exception_error:
        logger.error(f"Permission denied while reading file at: {path}")
        raise exception_error

    except IsADirectoryError as exception_error:
        logger.error(f"{exception_error}")
        raise exception_error
    
    except Exception as exception_error:
        # Catch-all for other exceptions
        logger.error(f"Unexpected error while encoding image at {path}: {exception_error}")
        raise exception_error
    
    
def convert_paths_to_str(obj: dict) -> dict:
    """
    Recursively convert Path objects in a nested dictionary to strings.
    """
    output = {}
    for key, value in obj.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, dict):
            output[key] = convert_paths_to_str(value)
        else:
            output[key] = value
    return output
