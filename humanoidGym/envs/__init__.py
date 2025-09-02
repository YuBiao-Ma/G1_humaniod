from humanoidGym import GYM_ROOT_DIR, GYM_ENVS_DIR

from .g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from .g1.g1_env import G1Robot

from .g1.g1_teacher_config import G1TeacherCfg,G1TeacherCfgPPO
from .g1.g1_teacher_env import G1TeacherRobot

from .long.long_uneven_config import LongUnevenRoughCfg,LongUnevenRoughCfgPPO
from .long.long_uneven_ori_config import LongOriUnevenRoughCfg,LongOriUnevenRoughCfgPPO

from .long.long_uneven_env import LongUnevenRobot

from .long.long_stand_config import LongStandRoughCfg,LongStandRoughCfgPPO
from .long.long_stand_env import LongStandRobot

from .long.long_uneven_ori_sop_config import LongOriSopUnevenRoughCfg,LongOriSopUnevenRoughCfgPPO

from .miniloong.miniloong_config import MiniloongCfg,MiniloongCfgPPO
from .miniloong.miniloong_env import MiniloongRobot

from .base.legged_robot import LeggedRobot

from humanoidGym.utils.task_registry import task_registry

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register( "g1_teacher", G1TeacherRobot,G1TeacherCfg(),G1TeacherCfgPPO())
task_registry.register("long_uneven",LongUnevenRobot,LongUnevenRoughCfg(),LongUnevenRoughCfgPPO())
task_registry.register("long_uneven_ori",LongUnevenRobot,LongOriUnevenRoughCfg(),LongOriUnevenRoughCfgPPO())
task_registry.register("long_uneven_ori_sop",LongUnevenRobot,LongOriSopUnevenRoughCfg(),LongOriSopUnevenRoughCfgPPO())
task_registry.register("long_stand",LongStandRobot,LongStandRoughCfg(),LongStandRoughCfgPPO())
task_registry.register("miniloong",MiniloongRobot,MiniloongCfg(),MiniloongCfgPPO())