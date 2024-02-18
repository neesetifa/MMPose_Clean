import os
import logging
import logging.config
from datetime import datetime

LOGGING_NAME = 'PoseEstimator'
def set_logging(name=LOGGING_NAME, log_file_path=None, verbose=True):
    if log_file_path is None:
        log_file_path = f'work_dirs/log_{datetime.now().strftime("%Y%m%d%H%M")}.txt'
        
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,

        # Define different formatters, they can be used in `handlers`
        'formatters': {'short': {'format': '%(message)s'},
                       'standard': {'format': '%(asctime)s %(levelname)s %(name)s: %(message)s'}
                       },

        # Define different handlers, names can be arbitrary, they can be used in `loggers`
        'handlers': {f'{name}_streamHandler': {'class': 'logging.StreamHandler',
                                               'formatter': 'short',
                                               'level': level},
                     f'{name}_fileHandler': {'class': 'logging.FileHandler',
                                             'formatter': 'standard',
                                             'level': level,
                                             'filename': log_file_path,
                                             'mode': 'a',  # a: add, w: overwrite
                                             'encoding': 'utf8'
                                             }
                     },

        # Define logger, you can define different kinds of logger with different set,
        # use logging.getLogger(`name`) to get specific logger
        'loggers': {name: {'level': level,
                           'handlers': [f'{name}_streamHandler', f'{name}_fileHandler'],
                           'propagate': False, },
                    'sjhfajkfka': {'level': level,
                                   'handlers': [ f'{name}_fileHandler'],
                                   'propagate': False, },
                    }
    })
    
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
