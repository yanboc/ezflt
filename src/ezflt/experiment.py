"""
实验管理模块

使用Hydra和Wandb管理多组实验，提供默认的实验管理功能。
"""

from typing import Optional, Dict, Any
from pathlib import Path


class ExperimentManager:
    """
    实验管理器
    
    使用Hydra进行配置管理，使用Wandb进行实验追踪。
    这是ezflt的默认实验管理方式。
    """
    
    def __init__(
        self,
        config_path: str = "configs",
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        use_wandb: bool = True,
        use_hydra: bool = True,
    ):
        """
        Args:
            config_path: Hydra配置文件路径
            project_name: Wandb项目名称
            experiment_name: 实验名称（如果为None，Hydra会自动生成）
            use_wandb: 是否使用Wandb（默认True）
            use_hydra: 是否使用Hydra（默认True）
        """
        self.config_path = config_path
        self.project_name = project_name or "ezflt-experiments"
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_hydra = use_hydra
        
        self.hydra_cfg = None
        self.wandb_run = None
        self._hydra_initialized = False
        self._wandb_initialized = False
    
    def _init_hydra(self):
        """初始化Hydra"""
        if not self.use_hydra:
            return
        
        if not self._hydra_initialized:
            try:
                from hydra import initialize_config_dir, compose
                from hydra.core.global_hydra import GlobalHydra
                
                # 如果已经初始化，先清理
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                
                # 获取配置文件目录的绝对路径
                config_dir = Path(self.config_path).absolute()
                if not config_dir.exists():
                    # 如果配置文件目录不存在，使用默认配置
                    print(f"警告: 配置文件目录 {config_dir} 不存在，将使用默认配置")
                    self.use_hydra = False
                    return
                
                # 初始化Hydra
                with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
                    self.hydra_cfg = compose(config_name="config")
                self._hydra_initialized = True
            except ImportError:
                raise ImportError(
                    "hydra-core未安装。请使用 'pip install hydra-core' 安装。"
                )
            except Exception as e:
                # 如果配置文件不存在，使用默认配置
                print(f"警告: Hydra初始化失败: {e}，将使用默认配置")
                self.use_hydra = False
    
    def _init_wandb(self, config: Optional[Dict[str, Any]] = None):
        """初始化Wandb"""
        if not self.use_wandb:
            return
        
        if not self._wandb_initialized:
            try:
                import wandb
                
                # 准备配置
                wandb_config = config or {}
                if self.hydra_cfg:
                    # 将Hydra配置转换为字典
                    wandb_config = self._hydra_to_dict(self.hydra_cfg)
                
                # 初始化wandb
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=wandb_config,
                    reinit=True,
                )
                self._wandb_initialized = True
            except ImportError:
                raise ImportError(
                    "wandb未安装。请使用 'pip install wandb' 安装。"
                )
    
    def _hydra_to_dict(self, cfg) -> Dict[str, Any]:
        """将Hydra配置对象转换为字典"""
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置（优先使用Hydra，否则返回空字典）"""
        if self.hydra_cfg:
            return self._hydra_to_dict(self.hydra_cfg)
        return {}
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标到Wandb"""
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
    
    def log_features(self, features: Dict[str, Any], epoch: int):
        """记录特征信息到Wandb"""
        if self.wandb_run:
            feature_metrics = {}
            for name, feature in features.items():
                if hasattr(feature, 'get_latest'):
                    latest = feature.get_latest()
                    if latest is not None:
                        feature_metrics[f"{name}/norm"] = latest.norm().item()
                        feature_metrics[f"{name}/mean"] = latest.mean().item()
                        feature_metrics[f"{name}/std"] = latest.std().item()
            
            if feature_metrics:
                self.log_metrics(feature_metrics, step=epoch)
    
    def finish(self):
        """完成实验，清理资源"""
        if self.wandb_run:
            self.wandb_run.finish()
    
    def __enter__(self):
        """上下文管理器：进入时初始化"""
        self._init_hydra()
        self._init_wandb()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器：退出时清理"""
        self.finish()


def create_experiment_manager(
    config_path: str = "configs",
    project_name: Optional[str] = None,
    use_wandb: bool = True,
    use_hydra: bool = True,
) -> ExperimentManager:
    """
    创建实验管理器（便捷函数）
    
    Args:
        config_path: Hydra配置文件路径
        project_name: Wandb项目名称
        use_wandb: 是否使用Wandb（默认True）
        use_hydra: 是否使用Hydra（默认True）
    
    Returns:
        ExperimentManager实例
    """
    return ExperimentManager(
        config_path=config_path,
        project_name=project_name,
        use_wandb=use_wandb,
        use_hydra=use_hydra,
    )

