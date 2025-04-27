import subprocess
import threading
from PyQt5.QtCore import QThread, pyqtSignal

class ProcessWorker(QThread):
    """Worker thread for running external processes without freezing the GUI"""
    output_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)
    process_finished = pyqtSignal(int)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None
        self.stopped = False
    
    def _read_output(self, pipe, signal):
        """Read from pipe and emit signal for each line"""
        while not self.stopped:
            line = pipe.readline()
            if not line:
                break
            signal.emit(line.rstrip())
    
    def run(self):
        """Run the command in a separate thread"""
        try:
            # Start the process with pipes for stdout/stderr
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Create threads to read stdout and stderr concurrently
            stdout_thread = threading.Thread(
                target=self._read_output,
                args=(self.process.stdout, self.output_ready)
            )
            stderr_thread = threading.Thread(
                target=self._read_output,
                args=(self.process.stderr, self.error_ready)
            )
            
            # Start reader threads
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            ret_code = self.process.wait()
            
            # Set stopped flag and wait for reader threads to finish
            self.stopped = True
            stdout_thread.join(1)  # Wait up to 1 sec
            stderr_thread.join(1)  # Wait up to 1 sec
            
            # Emit process finished signal with return code
            self.process_finished.emit(ret_code)
            
        except Exception as e:
            self.error_ready.emit(f"Error executing process: {str(e)}")
            self.process_finished.emit(-1)
