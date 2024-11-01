import os
import shutil
from typing import List, Tuple, Optional
from datetime import datetime

class FileOperations:
    """Handles basic file operations"""
    
    @staticmethod
    def copy_file(source: str, destination: str) -> bool:
        """
        Copy file from source to destination
        Returns: True if successful, False otherwise
        """
        try:
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            print(f"Error copying file: {str(e)}")
            return False

    @staticmethod
    def move_file(source: str, destination: str) -> bool:
        """
        Move file from source to destination
        Returns: True if successful, False otherwise
        """
        try:
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(source, destination)
            return True
        except Exception as e:
            print(f"Error moving file: {str(e)}")
            return False

    @staticmethod
    def delete_file(file_path: str) -> bool:
        """
        Delete file
        Returns: True if successful, False otherwise
        """
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False

    @staticmethod
    def rename_file(file_path: str, new_name: str) -> Optional[str]:
        """
        Rename file
        Returns: New file path if successful, None otherwise
        """
        try:
            directory = os.path.dirname(file_path)
            new_path = os.path.join(directory, new_name)
            os.rename(file_path, new_path)
            return new_path
        except Exception as e:
            print(f"Error renaming file: {str(e)}")
            return None

    @staticmethod
    def get_unique_filename(directory: str, filename: str) -> str:
        """Generate unique filename to avoid overwrites"""
        base_name, extension = os.path.splitext(filename)
        counter = 1
        new_filename = filename

        while os.path.exists(os.path.join(directory, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1

        return new_filename

    @staticmethod
    def create_directory(path: str) -> bool:
        """Create directory if it doesn't exist"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
            return False

    @staticmethod
    def list_files(directory: str, recursive: bool = False) -> List[str]:
        """List all files in directory"""
        files = []
        try:
            if recursive:
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
            else:
                files = [os.path.join(directory, f) for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f))]
        except Exception as e:
            print(f"Error listing files: {str(e)}")
        return files