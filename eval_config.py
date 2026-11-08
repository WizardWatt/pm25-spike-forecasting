"""Person B's view of the shared settings. Reads A's config.py and derives what the evaluation code
needs, so config.py stays the single source of truth and is never edited.
A's split boundaries are end-exclusive dates; here they become inclusive hourly bounds."""
import pandas as pd
import config as _c

STATIONS = list(_c.STATIONS)
PRIMARY_STATION, SECOND_STATION = STATIONS[0], STATIONS[1]
HORIZON = _c.HORIZON
SPIKE_THRESHOLD = _c.THRESHOLD
CLEAN_PATH = _c.CLEAN_PATH
FEATURES_PATH = _c.FEATURES_PATH


def _last_hour_before(d):
    return str(pd.Timestamp(d) - pd.Timedelta(hours=1))


TRAIN_START = "2000-01-01 00:00:00"                 # no lower bound: start of the data
TRAIN_END = _last_hour_before(_c.TRAIN_END)         # 2017-12-31 23:00:00
VAL_START, VAL_END = str(pd.Timestamp(_c.TRAIN_END)), _last_hour_before(_c.VAL_END)
REFIT_END = VAL_END                                  # refit on everything up to end of 2018
TEST_START, TEST_END = str(pd.Timestamp(_c.VAL_END)), _last_hour_before(_c.TEST_END)

LOCKDOWN_START = _c.LOCKDOWN_START
DIWALI_DATES, DIWALI_WINDOW_DAYS = list(_c.DIWALI_DATES), _c.DIWALI_WINDOW_DAYS

CLEAN_COLUMNS = ["Datetime", "StationId", "PM2.5", "y_reg", "y_spike"]
PRED_COLUMNS = ["Datetime", "StationId", "y_true", "y_pred", "lower", "upper", "spike_prob"]
