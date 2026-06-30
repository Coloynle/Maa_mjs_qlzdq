"""
赠礼选择 Agent — 状态管理
=======================
全局变量：
  _claimed_cat1: set[str]   — 已拿到信物并排除的武将名
  _claimed_cat2: set[str]   — 已拿到"驰援"并排除的武将名
  _current_general: str     — 当前轮命中的武将名
  _current_category: str    — "cat1" / "cat2"
"""

import json
import os

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

# 默认武将列表（当所有来源都为空时回退）
CAT1_DEFAULT = [
    "马超",
    "吕布",
    "关羽",
    "张春华",
    "黄忠",
    "司马懿",
    "孙策",
    "甘宁",
    "刘表",
]
CAT2_DEFAULT = ["刘禅", "曹丕", "鲁肃", "曹仁"]

# 本地配置文件的路径（与当前 py 同目录）
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "gift_config.json")

# ==================== 全局状态 ====================

_claimed_cat1: set[str] = set()
_claimed_cat2: set[str] = set()
_current_general: str = ""
_current_category: str = ""


# ==================== 辅助函数 ====================


def _read_config() -> dict | None:
    """读取 gift_config.json，失败或不存在时返回 None"""
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_list(raw: str, default: list[str]) -> list[str]:
    """
    把 "马超,吕布,关羽" 格式的逗号分隔字符串解析为列表
    raw 为空或非字符串时返回 default 的副本
    """
    if not raw or not isinstance(raw, str):
        return list(default)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items if items else list(default)


def _get_active_list(category: str, list_str: str = "") -> list[str]:
    """
    获取排除了已领取武将后的活跃列表（用于 OCR expected）。
    优先级：list_str（来自 UI） > gift_config.json > 代码默认列表
    """
    if list_str:
        # 优先级 1：用户通过 UI 输入的列表（最优先）
        full = _parse_list(
            list_str, CAT1_DEFAULT if category == "cat1" else CAT2_DEFAULT
        )
    else:
        # 优先级 2：从本地配置文件中读取
        config = _read_config()
        if config:
            key = "cat1_list" if category == "cat1" else "cat2_list"
            full = _parse_list(
                config.get(key, ""),
                CAT1_DEFAULT if category == "cat1" else CAT2_DEFAULT,
            )
        else:
            # 优先级 3：代码内置默认列表
            full = CAT1_DEFAULT if category == "cat1" else CAT2_DEFAULT

    # 排除已领取的武将
    claimed = _claimed_cat1 if category == "cat1" else _claimed_cat2
    return [n for n in full if n not in claimed]


def _ocr_active_general(context: Context, category: str, list_str: str = ""):
    """
    在底部武将选择栏 ([179, 509, 929, 49]) 做 OCR，
    返回最佳匹配的 OCRResult，没匹配到返回 None。

    参数：
      context  — 上下文，用于截图和调 OCR
      category — "cat1" / "cat2"
      list_str — 用户 UI 输入的逗号分隔列表（可选）

    ── context.run_recognition() 语法 ──
    context.run_recognition("临时节点名", image_numpy, pipeline_override_dict)
      → 第一个参数：临时节点名（随便起，仅 override 里的 key 需要对应）
      → 第二个参数：numpy.ndarray 格式的截图数据
      → 第三个参数：动态 pipeline 配置字典
      → 返回 RecognitionDetail | None

    RecognitionDetail:
      .hit: bool                       — 是否命中
      .best_result: OCRResult | None   — 最佳匹配
        .text: str                     — OCR 识别出的文本
        .box: [x, y, w, h]            — 命中框
        .score: float                  — 置信度

    ── context.tasker.controller.cached_image ──
    获取最近一次截图的 numpy 数组。如果从未截图会抛 RuntimeError。
    """
    active = _get_active_list(category, list_str)
    if not active:
        return None

    try:
        image = context.tasker.controller.cached_image
    except RuntimeError:
        return None

    # 主动调用 OCR 识别
    reco = context.run_recognition(
        "ocr",
        image,
        {
            "ocr": {
                "recognition": "OCR",  # 识别算法类型
                "roi": [179, 509, 929, 49],  # 识别区域 [x, y, w, h]
                "expected": active,  # 期望匹配的文本列表
                "order_by": "Expected",  # 按 expected 顺序优先匹配
            }
        },
    )

    if reco and reco.hit and reco.best_result and hasattr(reco.best_result, "text"):
        return reco.best_result
    return None


# ==================== CustomAction 注册语法说明 ====================
#
# @AgentServer.custom_action("动作名")
# class 类名(CustomAction):
#     def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
#         ...
#
# 装饰器把类注册为 pipeline 可调用的自定义动作。
# pipeline JSON 中通过 "custom_action": "动作名" 来引用。
#
# run() 参数：
#   context — 操作上下文
#   argv    — RunArg 参数包，包含：
#     .node_name: str               — 当前节点名
#     .custom_action_name: str      — 注册的动作名
#     .custom_action_param: str     — pipeline 中 custom_action_param 的值（JSON 字符串）
#     .reco_detail: RecognitionDetail — 前序识别结果
#     .box: Rect                    — 前序识别框 [x, y, w, h]
#
# 返回值：
#   True  → 动作成功，pipeline 走 next 列表
#   False → 动作失败，pipeline 走 on_error 或尝试 next 中下一个
# =================================================================


@AgentServer.custom_action("reset_gift_state")
class ResetGiftState(CustomAction):
    """
    重置赠礼状态 — 每次千里开局选将后执行一次。
    pipeline 中配 DirectHit + Custom，不依赖任何前序识别结果。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _claimed_cat1, _claimed_cat2, _current_general, _current_category
        _claimed_cat1.clear()
        _claimed_cat2.clear()
        _current_general = ""
        _current_category = ""
        return True


@AgentServer.custom_action("pick_and_record")
class PickAndRecord(CustomAction):
    """
    记录 OCR 命中的武将名到全局状态，不执行点击。
    点击由后续的 click_general 节点负责。

    读取 argv.custom_action_param 中的 category 和 list：
      custom_action_param: '{"category": "cat1", "list": "{cat1_list}"}'
      → category 固定写死，list 由 UI 输入插值替换
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _current_general, _current_category

        # argv.custom_action_param 是 JSON 字符串，由 pipeline 配置
        params = (
            json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        )
        category = params.get("category", "")  # "cat1" / "cat2"
        list_str = params.get("list", "")  # UI 输入的逗号分隔列表（可能为空）

        if category not in ("cat1", "cat2"):
            return False

        result = _ocr_active_general(context, category, list_str)
        if not result:
            return False  # 没匹配到任何活跃武将

        # result.text = OCR 识别出的文本，就是武将名
        _current_general = result.text
        _current_category = category
        return True


@AgentServer.custom_action("click_general")
class ClickGeneral(CustomAction):
    """
    在底栏 OCR 匹配武将并点击。
    与 pick_and_record 使用相同的 category + list 参数，
    确保点的是同一个武将（同一张截图，结果应一致）。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        params = (
            json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        )
        category = params.get("category", "")
        list_str = params.get("list", "")

        if category not in ("cat1", "cat2"):
            return False

        result = _ocr_active_general(context, category, list_str)
        if not result:
            return False

        # result.box = [x, y, w, h]，计算中心点坐标
        box = result.box
        if not box or len(box) != 4:
            return False
        cx = box[0] + box[2] // 2
        cy = box[1] + box[3] // 2

        # post_click() 是异步操作，返回一个 Job
        # .wait() 阻塞等待点击完成
        context.tasker.controller.post_click(cx, cy).wait()
        return True


@AgentServer.custom_action("record_item")
class RecordItem(CustomAction):
    """
    驰援排序后的物品判断：
    OCR 物品选择区域，检查是否出现了"驰援"。
    如果不是"驰援" → 释放武将（_current_general 置空），
    后续 mark_general_claimed 就不会将其加入排除集。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _current_general

        # 非 cat2 路径不需要判断，直接放过
        if _current_category != "cat2":
            return True

        try:
            image = context.tasker.controller.cached_image
        except RuntimeError:
            return False

        # OCR 物品选择区域，只检查"驰援"是否出现
        reco = context.run_recognition(
            "ocr_item",
            image,
            {
                "ocr_item": {
                    "recognition": "OCR",
                    "roi": [522, 171, 171, 377],
                    "expected": ["驰援"],
                }
            },
        )

        # 没匹配到"驰援" → cat2 武将没拿到目标物品 → 不放回排除集
        if not (reco and reco.hit):
            _current_general = ""

        return True


@AgentServer.custom_action("mark_general_claimed")
class MarkGeneralClaimed(CustomAction):
    """
    标记武将已领取并加入排除集：
      cat1 → 直接加入 _claimed_cat1
      cat2 → 仅当 _current_general 有值（即 record_item 确认拿到"驰援"）才加入
    最后清空临时状态，为下一轮弹窗做准备。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        global _current_general, _current_category

        if _current_category == "cat1" and _current_general:
            _claimed_cat1.add(_current_general)

        if _current_category == "cat2" and _current_general:
            _claimed_cat2.add(_current_general)

        # 清空临时状态，下次弹窗重新记录
        _current_general = ""
        _current_category = ""

        return True
