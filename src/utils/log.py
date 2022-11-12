import json, os, shutil
import numpy as np
import torch as tc
from datetime import datetime
from addict import Dict
from filelock import FileLock

RECORD = Dict({'model' : ['arch', 'checkpoint', 'width', 'depth', 'activation'],
          'dataset' : ['dataset', 'idbh'],
          'training' : ['optim', 'lr', 'batch_size', 'annealing', 'momentum', 'weight_decay', 'swa']})

BASE = 36
ID_LENGTH = 4

def complete_ids(ids):
    while(len(ids) < ID_LENGTH):
        ids = '0' + ids
    return ids

def ids_from_idx(idx):
    try:
        ids = np.base_repr(idx, BASE).lower()
        ids = complete_ids(ids)
        return ids
    except:
        raise Exception("Invalid index: {}".format(idx))

def idx_from_ids(ids):
    try:
        return int(ids, BASE)
    except:
        raise Exception("Invalid id string: {}".format(ids))
    
class Logger:
    def __init__(self, log_filepath, info=None):
        self.log_filepath = log_filepath
        self.new_log = None
        self.log_id = None
        self.change = None
        
        if os.path.isfile(log_filepath):
            self.lock.acquire()
            with open(log_filepath, 'r') as f:
                self.logbook = [Dict(log) for log in json.load(f)]
            self.lock.release()
        else:
            # no existing log file found
            self.logbook = []
        if info:
            self.new(info)
            
    def __len__(self):
        return len(self.logbook)

    def __getitem__(self, key):
        if isinstance(key, str):
            for log in self.logbook:                
                if log.id == key:
                    return log
            return None
        else:
            raise Exception("Invalid key for accessing logbook: {}".format(key))
                
    def __setitem__(self, ids, val):
        for i, log in enumerate(self.logbook):
            if log.id == ids:
                self.logbook[i] = val
                
    def refresh(self):
        if os.path.isfile(self.log_filepath):
            backup_file = self.log_filepath + '~'
            shutil.copyfile(self.log_filepath, backup_file)
            with open(self.log_filepath, 'r') as f:
                self.logbook = [Dict(log) for log in json.load(f)]
                
    def new_id(self):
        if len(self.logbook) > 0:
            return idx_from_ids(self.logbook[-1].id) + 1
        else:
            return 0
                
    @property
    def size(self):
        return len(self)

    @property
    def lock(self):
        if not hasattr(self, '_lock'):
            self._lock = FileLock(self.log_filepath + '.lock')
        return self._lock
            
    @property
    def time(self):
        return datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
    def new(self, info):
        record = Dict({'id' : None,
                       'tmp_id' : info.tmp_id,
                       'job_id' : info.job_id,
                       'abstract' : info.abstract(),
                       'checkpoint' : {},
                       'robustness' : {},
                       'status' : 'normal'})
        if record.job_id is None:
            del record['job_id']

        if info.advt:
            RECORD.training += ['eps', 'eps_step', 'max_iter', 'warm_start']
            
        for head, attrs in RECORD.items():
            for attr in attrs:
                if hasattr(info, attr):
                    record[head][attr] = getattr(info, attr)

        self.new_log = record
        return self.new_log

    def fetch(self, ids):
        return self[ids]
    
    def update(self, *heads, ids=None, save=False, **kwargs):
        if ids is None:
            if self.new_log is None:
                return self.update(*heads, ids=self.log_id, save=save, **kwargs)
            else:
                log = self.new_log
        else:
            if self.change is None:
                self.change = Dict()
            if ids not in self.change:
                self.change[ids] = Dict()
            log = self.change[ids]
            
        if heads is not None:
            for head in heads:
                if head not in log:
                    log[head] = Dict()
                log = log[head]

        if kwargs is None:
            raise Exception("Void update for the head {} of log {}".format(heads, ids))
        merge(log, kwargs)
        
        if save:
            self.save(False)

    def save(self, report=True):
        if not valid_new_log(self.new_log) and self.change is None:
            return
        else:
            self.lock.acquire()

            self.refresh()
                
            if valid_new_log(self.new_log):
                self.new_log.time.create = self.time
                self.new_log.id = ids_from_idx(self.new_id())
                self.log_id = self.new_log.id
                self.logbook.append(self.new_log)
                self.new_log = None
                
            if self.change is not None:
                for ids, change in self.change.items():
                    log = self[ids]
                    merge(log, change)
                    log.time.modify = self.time
                    self[ids] = log
                self.change = None
                
            with open(self.log_filepath, 'w') as f:
                json.dump(self.logbook, f, indent=4)

            self.lock.release()
            if report:
                print("Logbook saved successfully.")

def valid_new_log(log):
    # only record the new log with valid accuracy
    return log is not None and len(log.checkpoint)>1

def merge(dict1, dict2):
    for k, v in dict2.items():
        if k in dict1:
            _v = dict1[k]
            if isinstance(_v, dict) or isinstance(_v, Dict):
                merge(_v, v)
            else:
                dict1[k] = v
        else:
            dict1[k] = v
