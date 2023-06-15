import sys
import json
import shutil
from pathlib import Path
from datetime import datetime


class SavePaths(object):
    """
    A class to generate save paths
    """

    _CONFIG = json.load(open(f"{Path(__file__).parent}/config.json", "r"))

    def __init__(self, name: str):
        """
        Parameters:
            name (str): name of the run
        """
        root_reg = datetime.now().strftime(f"{name}/%Y-%m-%d_%Hh-%Mm-%Ss")

        self.exec_path = Path(__file__).parent.parent.parent

        self.current_run_folder = self.exec_path / "runs" / root_reg
        self.current_run_folder.mkdir(parents=True, exist_ok=True)

        self.log = Path(self.current_run_folder) / "logs"
        self.log.mkdir(parents=True, exist_ok=True)

        with open(self.log / "config.json.bak", "w") as f:
            json.dump(self._CONFIG, f, indent=4)

        sys.stdout = Logger(sys.stdout, self.log / "stdout.txt")
        sys.stderr = Logger(sys.stderr, self.log / "stderr.txt")

        self.result = Path(self.current_run_folder) / "results"
        self.result.mkdir(parents=True, exist_ok=True)

    def logging(self, text: str):
        """
        Log a text in the log file

        Parameters:
            text (str): text to log
        """
        with open(self.log / "log.txt", "a") as f:
            f.write(text + "\n")

    def move_result_to_final_path(self):
        """
        Move the result folder to the final path
        """
        shutil.copytree(self.result, self.exec_path / "results", dirs_exist_ok=True)


class Logger(object):
    """A class to log result in console and a file"""

    def __init__(self, terminal_ouput, filename_output_loc) -> None:
        """
        Parameters:
            terminal_ouput (str): terminal output
            filename_output_loc (str): file output
        """
        self.terminal_ouput = terminal_ouput
        self.file_output = open(filename_output_loc, "a")

    def write(self, message):
        """
        Write the message in the terminal and the file

        Parameters:
            message (str): message to write
        """
        self.terminal_ouput.write(message)
        self.file_output.write(message)

    def flush(self):
        """
        Flush the terminal and the file
        """
        self.terminal_ouput.flush()
        self.file_output.flush()
