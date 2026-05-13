# GPT-Image-2 生图插件

基于 `gpt-image-2` 兼容接口的 AstrBot 生图插件，支持文生图、图生图、异步任务轮询、OpenAI/NewAPI 风格同步 `b64_json` 返回、任务查询，以及 `1k` / `2k` / `4k` 分辨率档位。

仓库地址：<https://github.com/xxuanzQAQ/astrbot_plugin_gpt_image_2>

APIMart 注册入口：<https://apimart.ai/register?aff=J3ZjCO>

## 功能

- 文生图：通过文字提示词生成图片。
- 图生图：支持消息内图片、引用图片、图片 URL 和 `base64://` 图片作为参考图。
- 自动轮询：提交异步任务后可自动等待结果并发送图片。
- 同步结果：兼容 `data[0].b64_json` 这类同步返回图片数据的接口。
- 任务查询：可通过任务 ID 查询生成状态。
- 参数控制：支持比例、像素尺寸、分辨率档位、等待模式和官方渠道兜底。

## 安装

进入 AstrBot 插件目录后克隆本仓库：

```bash
cd /path/to/AstrBot/data/plugins
git clone https://github.com/xxuanzQAQ/astrbot_plugin_gpt_image_2.git
```

重启 AstrBot，随后在 WebUI 的插件管理中启用插件并填写配置。

如果依赖没有被自动安装，请在 AstrBot 的 Python 环境中安装：

```bash
pip install -r data/plugins/astrbot_plugin_gpt_image_2/requirements.txt
```

## 配置

在 AstrBot 插件配置中填写：

| 配置项 | 说明 | 默认值 |
| --- | --- | --- |
| `api_base_url` | API 地址；兼容 APIMart、OpenAI/NewAPI 风格 `/v1` 地址和完整接口地址 | `https://api.apimart.ai/v1` |
| `api_key` | API Key，请只保存在本地 AstrBot 配置中 | 空 |
| `model` | 模型名称 | `gpt-image-2` |
| `default_size` | 默认图片比例或像素尺寸 | `1:1` |
| `default_resolution` | 默认输出分辨率档位 | `1k` |
| `include_resolution` | 请求中是否附带 `resolution` 字段 | `true` |
| `official_fallback` | 是否启用官方渠道兜底 | `false` |
| `auto_wait` | 提交后是否自动等待并发送图片 | `true` |
| `initial_delay` | 提交任务后首次查询延迟，单位秒 | `12` |
| `poll_interval` | 轮询间隔，单位秒 | `5` |
| `poll_timeout` | 自动等待结果的最长时间，单位秒 | `180` |
| `request_timeout` | 单次 HTTP 请求超时，单位秒 | `60` |
| `transient_retries` | 502/503/504 等临时上游错误重试次数 | `2` |
| `transient_retry_delay` | 临时上游错误重试间隔，单位秒 | `5` |
| `max_reference_images` | 最多参考图数量，上限为 16 | `16` |
| `include_result_link` | 发送图片时是否附带结果 URL | `true` |

APIMart 的 `api_key` 获取入口：<https://apimart.ai/register?aff=J3ZjCO>

使用 NewAPI 兼容站点时，可将 `api_base_url` 配成站点的 `/v1` 地址，例如：

```text
https://newapi-hk.qianye.host/v1
```

插件会自动适配常见兼容接口参数：OpenAI 风格模型会自动使用像素尺寸并省略 `resolution`；如果接口返回 `resolution` 参数错误，会自动去掉该字段重试；如果默认 `gpt-image-2` 通道报 `chatgpt upstream 401: chat-requirements failed`，会自动尝试 `gpt-image-1` 兼容参数；如果遇到 502/503/504、`context deadline exceeded` 或 `poll error`，会短间隔自动重试。若重试后仍失败，说明接口站点的上游账号/Cookie、模型通道或网关超时时间本身不可用，需要在站点侧处理。

## 命令

### 文生图

```text
/gpt生图 一只橘猫坐在窗台上看夕阳，水彩画风格 --size 16:9 --resolution 2k
/gptimage 生成 星空下的古老城堡 --size 16:9 --resolution 4k
```

### 图生图

直接发送或引用图片，并附带命令：

```text
/gpt生图 把这张照片变成水彩画风格 --size 4:3 --resolution 2k
```

也可以显式传参考图 URL：

```text
/gpt生图 把两张照片融合成海报 --images https://example.com/a.png,https://example.com/b.png --size 4:3
```

### 查询任务

```text
/gptimage 查询 task_xxx
```

### 查看帮助

```text
/gptimage 帮助
```

## 命令参数

| 参数 | 说明 |
| --- | --- |
| `--size` | `auto`、比例值或像素尺寸，如 `16:9` / `9:16` / `1881x836` |
| `--resolution` | 输出分辨率档位：`1k` / `2k` / `4k` |
| `--model` | 临时指定模型名称 |
| `--include-resolution` | 是否附带 `resolution` 字段，`true` / `false` |
| `--no-resolution` | 是否不附带 `resolution` 字段，`true` / `false` |
| `--images` | 参考图 URL，多个用逗号、中文逗号或空格分隔 |
| `--wait` | 是否等待任务完成，`true` / `false` |
| `--no-wait` | 是否不等待任务完成，`true` / `false` |
| `--official-fallback` | 是否启用官方渠道兜底，`true` / `false` |

支持的比例：`auto`、`1:1`、`3:2`、`2:3`、`4:3`、`3:4`、`5:4`、`4:5`、`16:9`、`9:16`、`2:1`、`1:2`、`3:1`、`1:3`、`21:9`、`9:21`。

`4k` 仅支持 `16:9` / `9:16` / `2:1` / `1:2` / `3:1` / `1:3` / `21:9` / `9:21`。

## 隐私与安全

- 不要把 APIMart API Key 写入代码、README、Issue、截图或提交记录；只在 AstrBot 插件配置中填写。
- 插件本身不会把提示词、参考图或生成结果写入本仓库文件，但请求会发送到你配置的 `api_base_url`。
- 使用图生图时，消息图片、引用图片、图片 URL 或 base64 图片会作为参考图提交给你配置的接口服务，请避免处理不应上传到第三方服务的敏感图片。
- `include_result_link` 开启时，生成结果 URL 会随消息发送到聊天中；在群聊或公开频道中可按需关闭。
- 上传 GitHub 前请确认 `.env`、日志、数据库、运行数据、缓存目录和本地代理配置没有被加入暂存区。
- 如果 API Key 曾经出现在公开仓库、日志或聊天记录中，请立即在对应服务后台重新生成或撤销旧 Key。

## 开发

本插件主要文件：

| 文件 | 说明 |
| --- | --- |
| `main.py` | 插件主体逻辑 |
| `_conf_schema.json` | AstrBot 插件配置 schema |
| `metadata.yaml` | 插件元信息 |
| `requirements.txt` | Python 依赖 |
