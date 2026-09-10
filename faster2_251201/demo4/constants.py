from pathlib import Path


DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent.parent)
DEFAULT_MODEL_NAME = "D:/RWKV Runner/models/rwkv7-g1g-1.5b-20260526-ctx8192"
DEFAULT_PROMPT = "User: 生成HTML，创建一个温馨的本地咖啡馆官网，采用暖色调配色方案，包括浅米色、奶油白和柔和的木质色系，搭配优雅的排版风格，使用衬线字体和适度的间距。首页包含品牌介绍、菜单展示、活动信息和联系方式等模块，页面内容简洁明了，突出咖啡馆的特色和氛围\n\nAssistant: <think></think"
PROMPT_PRESETS = [
    "User: 生成网页：\n奇绩创坛是成立于2018年的投资机构与创业孵化平台，前身为YC中国，由陆奇创立，总部位于中国北京，系北京市标杆孵化器。机构专注早期科技项目投资，通过提供30万美元资金换取7%股权，并开设为期3个月的创业加速营，涵盖产品迭代、融资对接及战略规划等服务。其导师团队包括陆奇、黄峥、王怀南、邢波、雷鸣、栾运明、曹勖文等不同技术和创业领域的专业人士。截至2025年，累计投资加速500余家初创公司，总估值达900亿元，覆盖人工智能、机器人、大模型、量子计算等38个前沿技术领域。\n\nAssistant: <think></think",
    "User: 生成网页：\n天际资本（FutureX Capital）是一家专注于中美人工智能（AI）及硬科技生态的全球化双币种风险投资机构。\n核心概况\n创立背景：于2018年在香港创立，由前华夏基金私募股权业务创始人张倩（Cynthia Zhang）主导成立。\n监管资质：持有香港证监会（SFC）第4类（就证券提供意见）及第9类（资产管理）牌照。\n投资布局：致力于从 Pre-A 到 Pre-IPO 阶段的投资，主要聚焦核心赛道：AI 原生应用与基础设施新能源与高端智能汽车（如固态电池、碳化硅、自动驾驶芯片等）半导体与深科技（如异构集成、医疗器械、机器人等）\n代表投资案例\n天际资本在中美AI与科技生态中颇具影响力，曾投资包括字节跳动 (ByteDance)、PingCAP、Dify、Mistral AI、蔚来能源 (NIO Power)、黑芝麻智能等多家行业代表性企业。\n\nAssistant: <think></think",
    "User: 生成网页：UGREEN绿联是全球领先的消费电子科技品牌，业务覆盖180多个国家和地区，依托于充电创意、智能办公、智能影音、智能存储四大品类，为全球用户带来从容掌控的数字生活体验。\n\nAssistant: <think></think",
    "User: 生成HTML，创建一个温馨的本地咖啡馆官网，采用暖色调配色方案，包括浅米色、奶油白和柔和的木质色系，搭配优雅的排版风格，使用衬线字体和适度的间距。首页包含品牌介绍、菜单展示、活动信息和联系方式等模块，页面内容简洁明了，突出咖啡馆的特色和氛围\n\nAssistant: <think></think",
    "User: Write HTML: 3D animation of cars in forest with animals\n\nAssistant: <think></think",
    "User: Write HTML: 3D animation of a SpaceX rocket landing on Mars\n\nAssistant: <think></think",
    "User: Write HTML: interactive weather map with animated clouds and rain\n\nAssistant: <think></think",
    "User: Write HTML: retro arcade RPG start screen\n\nAssistant: <think></think",
    "User: Write HTML: storybook scene with a dragon flying over a castle\n\nAssistant: <think></think",
    "User: Write HTML: interactive dashboard for a city traffic system\n\nAssistant: <think></think",
    "User: Write HTML: animated aquarium with colorful fish and coral\n\nAssistant: <think></think",
    "User: Write HTML: sci-fi spaceship navigation interface\n\nAssistant: <think></think",
    "User: Write HTML: cozy cafe menu with animated steam and pastries\n\nAssistant: <think></think",
    "User: Write HTML: character sheet for a high fantasy RPG\n\nAssistant: <think></think",
    "User: Write HTML: a fancy hotel homepage\n\nAssistant: <think></think",
]
TITLE_MODEL_NAME = "RWKV-7 7.2B"
TITLE_PRECISION = "FP16"
TITLE_GPU_NAME = "RTX 5090"
WINDOW_TITLE = "RWKV-7 batch demo4"
