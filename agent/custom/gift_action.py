"""
赠礼选择 Agent — 状态管理
=======================
全局变量：
  _claimed_cat1: set[str]   — 已拿到信物并排除的武将名
  _claimed_cat2: set[str]   — 已拿到"驰援"并排除的武将名
"""

import json
import os
import time

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


def _click_and_wait(context, box):
    cx = box[0] + box[2] // 2
    cy = box[1] + box[3] // 2
    context.tasker.controller.post_click(cx, cy).wait()


def _poll_ocr(context, roi, expected, timeout=6.0, interval=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        context.tasker.controller.post_screencap().wait()
        try:
            image = context.tasker.controller.cached_image
        except RuntimeError:
            time.sleep(interval)
            continue
        reco = context.run_recognition(
            "_poll",
            image,
            {
                "_poll": {
                    "recognition": "OCR",
                    "roi": roi,
                    "expected": expected,
                    "order_by": "Expected",
                }
            },
        )
        if reco and reco.hit and reco.best_result:
            return reco.best_result
        time.sleep(interval)
    return None


@AgentServer.custom_action("handle_gift_selection")
class HandleGiftSelection(CustomAction):
    def run(self, context, argv):
        params = (
            json.loads(argv.custom_action_param)
            if isinstance(argv.custom_action_param, str)
            else (argv.custom_action_param or {})
        )
        cat1_list = params.get("cat1_list", "")
        cat2_list = params.get("cat2_list", "")

        context.tasker.controller.post_screencap().wait()
        result = _ocr_active_general(context, "cat1", cat1_list)
        if result:
            _click_and_wait(context, result.box)
            item = _poll_ocr(context, [522, 171, 171, 377], ["信物"])
            if item:
                _click_and_wait(context, item.box)
                _claimed_cat1.add(result.text)
            time.sleep(0.5)
            return True

        context.tasker.controller.post_screencap().wait()
        result = _ocr_active_general(context, "cat2", cat2_list)
        if result:
            _click_and_wait(context, result.box)
            item = _poll_ocr(context, [522, 171, 171, 377], ["驰援"])
            if item:
                _click_and_wait(context, item.box)
                _claimed_cat2.add(result.text)
            else:
                fallback = _poll_ocr(
                    context, [522, 171, 171, 377],
                    ["资助", "武将牌", "信物", "并肩作战"], timeout=2
                )
                if fallback:
                    _click_and_wait(context, fallback.box)
            time.sleep(0.5)
            return True

        fallback = _poll_ocr(context, [531, 504, 220, 51], ["赠礼"], timeout=3)
        if fallback:
            _click_and_wait(context, fallback.box)
            sort = _poll_ocr(context, [522, 171, 171, 377],
                             ["资助", "武将牌", "驰援", "信物", "并肩作战"], timeout=3)
            if sort:
                _click_and_wait(context, sort.box)
                time.sleep(0.5)
        return True


@AgentServer.custom_action("reset_gift_state")
class ResetGiftState(CustomAction):
    """
    重置赠礼状态 — 每次千里开局选将后执行一次。
    pipeline 中配 DirectHit + Custom，不依赖任何前序识别结果。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        _claimed_cat1.clear()
        _claimed_cat2.clear()
        return True
