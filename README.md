# GPT-Image-2 生图插件

基于 `gpt-image-2` 兼容接口的 AstrBot 生图插件，支持文生图、图生图、异步任务轮询、OpenAI/NewAPI 风格同步 `b64_json` 返回、`/chat/completions` 图片模型、任务查询，以及 `1k` / `2k` / `4k` 分辨率档位。

仓库地址：<https://github.com/xxuanzQAQ/astrbot_plugin_gpt_image_2>

APIMart 注册入口：<https://apimart.ai/register?aff=J3ZjCO>

## 功能

- 文生图：通过文字提示词生成图片。
- 图生图：支持消息内图片、引用图片、图片 URL 和 `base64://` 图片作为参考图。
- 自动轮询：提交异步任务后可自动等待结果并发送图片。
- 同步结果：兼容 `data[0].b64_json` 这类同步返回图片数据的接口。
- Chat Completions：兼容 `gemini-3.1-flash-image` 等 `/v1/chat/completions` 图片模型。
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
| `api_mode` | 接口模式：`auto`、`images`、`chat` | `auto` |
| `chat_temperature` | `/chat/completions` 的 temperature | `0.7` |
| `auto_wait` | 提交后是否自动等待并发送图片 | `true` |
| `initial_delay` | 提交任务后首次查询延迟，单位秒 | `12` |
| `poll_interval` | 轮询间隔，单位秒 | `5` |
| `poll_timeout` | 自动等待结果的最长时间，单位秒 | `180` |
| `request_timeout` | 单次 HTTP 请求超时，单位秒 | `60` |
| `transient_retries` | 502/503/504 等临时上游错误重试次数 | `2` |
| `transient_retry_delay` | 临时上游错误重试间隔，单位秒 | `5` |
| `newapi_log_lookup` | NewAPI 同步提交 504 后是否查询后台日志并返回最终错误 | `true` |
| `newapi_log_key` | NewAPI 日志接口鉴权 Key；留空默认尝试使用 `api_key` | 空 |
| `newapi_log_lookup_timeout` | 504 后等待后台日志落库的最长秒数 | `45` |
| `newapi_log_lookup_interval` | 504 后查询后台日志的间隔秒数 | `5` |
| `max_reference_images` | 最多参考图数量，上限为 16 | `16` |
| `include_result_link` | 发送图片时是否附带结果 URL | `true` |
| `debug_log_payload` | 是否记录脱敏后的请求 payload、响应和参考图提交值 | `false` |
| `image_to_image_endpoint` | 图生图接口路径；`auto`、`/images/generations` 或 `/images/edits` | `auto` |

APIMart 的 `api_key` 获取入口：<https://apimart.ai/register?aff=J3ZjCO>

使用 NewAPI 兼容站点时，可将 `api_base_url` 配成站点的 `/v1` 地址，例如：

```text
https://newapi-hk.qianye.host/v1
```

使用 NewAPI Chat Completions 图片模型时，可将 `api_base_url` 配成 `/v1` 或完整 `/v1/chat/completions` 地址，并通过 `--model` 临时切换模型：

```text
/gpt生图 一只橘猫坐在窗台上看夕阳 --model gemini-3.1-flash-image --api-mode chat
```

插件会自动适配常见兼容接口参数：OpenAI 风格模型会自动使用像素尺寸并省略 `resolution`；消息内图片和引用图片会优先转成 base64 data URI 后提交，避免平台临时图片 URL 无法被接口服务访问；图生图在 APIMart 默认继续走 `/images/generations + image_urls`，在其他兼容站点默认走 `/images/edits` 并用 multipart 上传图片；如果接口返回 `resolution` 参数错误，会自动去掉该字段重试；如果默认 `gpt-image-2` 通道报 `chatgpt upstream 401: chat-requirements failed`，会自动尝试 `gpt-image-1` 兼容参数；如果遇到 502/503/504、`context deadline exceeded` 或 `poll error`，会短间隔自动重试。错误信息会按 API 接口错误、模型配置错误、网络异常、内容安全审核拦截分类展示；当网关错误页面中包含明确的上游业务错误正文时，会优先原样返回上游报错；如果 NewAPI 同步提交阶段被网关 504 中断，插件会尝试查询 NewAPI 后台日志并返回稍后落库的最终上游错误。若重试后仍失败，说明接口站点的上游账号/Cookie、模型通道或网关超时时间本身不可用，需要在站点侧处理。

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
| `--api-mode` | 临时指定接口模式：`images` / `chat` |
| `--temperature` | Chat Completions 模式的 temperature |
| `--include-resolution` | 是否附带 `resolution` 字段，`true` / `false` |
| `--no-resolution` | 是否不附带 `resolution` 字段，`true` / `false` |
| `--images` | 参考图 URL，多个用逗号、中文逗号或空格分隔 |
| `--wait` | 是否等待任务完成，`true` / `false` |
| `--no-wait` | 是否不等待任务完成，`true` / `false` |
| `--official-fallback` | 是否启用官方渠道兜底，`true` / `false` |

支持的比例：`auto`、`1:1`、`3:2`、`2:3`、`4:3`、`3:4`、`5:4`、`4:5`、`16:9`、`9:16`、`2:1`、`1:2`、`3:1`、`1:3`、`21:9`、`9:21`。

`4k` 仅支持 `16:9` / `9:16` / `2:1` / `1:2` / `3:1` / `1:3` / `21:9` / `9:21`。

## 开发

本插件主要文件：

| 文件 | 说明 |
| --- | --- |
| `main.py` | 插件主体逻辑 |
| `_conf_schema.json` | AstrBot 插件配置 schema |
| `metadata.yaml` | 插件元信息 |
| `requirements.txt` | Python 依赖 |
