import logging
import os
import sys
from iopath.common.file_io import g_pathmgr
import atexit
import functools
from typing import Any, Dict, Optional, Union

import wandb
from numpy import ndarray
from torch import Tensor
from hydra.utils import instantiate

from training.utils.train_utils import get_machine_local_and_dist_rank, makedir



Scalar = Union[Tensor, ndarray, float, int]

class WandbLogger:
    def __init__(self,
                    project: str = 'BrainSAM',
                    name: Optional[str] = None,
                    config: Optional[Dict[str, Any]] = None,
                    log_dir: str = './wandb_logs',
                    **kwargs
                 )-> None:
        self._run = None
        _, self._rank = get_machine_local_and_dist_rank()

        if self._rank == 0:
            makedir(log_dir)
            logging.info(f"WandbLogger initialized at {log_dir}. ")
            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                dir=log_dir,
                resume="allow",
                **kwargs
            )
        else:
            logging.debug(f"WandbLogger disabled on rank {self._rank}.")

    @property
    def run(self):
        return self._run

    def log_dict(self,payload: Dict[str, Scalar], step: int)-> None:
        if self._run:
            self._run.log(payload, step=step)

    def log(self, name:str, data: Scalar, step: int)-> None:
        """记录单个指标."""
        if self._run:
            self._run.log({name: data}, step=step)

    def log_hparams(self, hparams: Dict[str, Any], meters: Optional[Dict[str, Scalar]] = None)-> None:
        """记录超参数和指标."""
        if self._run:
            wandb.config.update(hparams, allow_val_change=True)
            if meters:
                self._run.log(meters)

    def close(self)-> None:
        """关闭WandbLogger."""
        if self._run:
            self._run.finish()
            self._run = None
            logging.info("WandbLogger closed.")

class Logger:
    """
    统一的日志记录接口，支持通过yaml初始化。
    """
    def __init__(self,logging_conf):
        wb_config = logging_conf.wandb_writer
        wb_should_log = wb_config and wb_config.pop('should_log', True)

        if wb_should_log:
            self.wb_logger = instantiate(wb_config)
        else:
            self.wb_logger = None

    def log_dict(self,payload: Dict[str, Scalar], step: int)-> None:
        if self.wb_logger:
            self.wb_logger.log_dict(payload, step)

    def log(self, name:str, data: Scalar, step: int)-> None:
        if self.wb_logger:
            self.wb_logger.log(name, data, step)

    def log_hparams(self, hparams: Dict[str, Any], meters: Optional[Dict[str, Scalar]] = None)-> None:
        if self.wb_logger:
            self.wb_logger.log_hparams(hparams, meters)


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    """缓存文件流，确保多个 logger 能安全写入同一个文件"""
    log_buffer_kb = 10 * 1024  # 10KB 缓存
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
        name,
        output_dir=None,
        rank=0,
        log_level_primary="INFO",
        log_level_secondary="ERROR",
):

    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # 清理旧处理器
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 分布式处理：只有 rank 0 打印 INFO，其他只打 ERROR
    if rank == 0:
        console_handler.setLevel(log_level_primary)
        if log_filename:
            file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
            file_handler.setLevel(log_level_primary)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        console_handler.setLevel(log_level_secondary)

    logging.root = logger


def shutdown_logging():
    logging.info("Shutting down loggers...")
    for handler in logging.root.handlers:
        handler.close()