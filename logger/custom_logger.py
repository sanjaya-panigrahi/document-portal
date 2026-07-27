from datetime import datetime
import logging
import os


class CustomLogger:
    def __init__(self):
        # Set up the logs directory
        self.logs_dir = "logs"
        self.logs_dir = os.path.join(os.getcwd(), self.logs_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create log file with timestamp
        log_file = f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)
        
        #Configure Logging
        logging.basicConfig(
            filename= self.log_file_path,
            format= "[%(asctime)s] %(levelname)s %(name)s (line:%(lineno)d) - %(message)s",
            level= logging.INFO,
)

    def get_logger(self, name=__file__):
        return logging.getLogger(os.path.basename(name))
    
if __name__ == "__main__":
    custom_logger = CustomLogger()
    logger = custom_logger.get_logger(__file__)
    logger.info("Logging is set up and ready to go!")