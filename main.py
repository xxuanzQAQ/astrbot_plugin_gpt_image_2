import asyncio
import base64
import json
import re
from typing import Any

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_gpt_image_2"
REGISTER_URL = "https://apimart.ai/register?aff=J3ZjCO"

SUPPORTED_SIZES = {
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "16:9",
    "9:16",
    "2:1",
    "1:2",
    "3:1",
    "1:3",
    "21:9",
    "9:21",
}
FOUR_K_SIZES = {"16:9", "9:16", "2:1", "1:2", "3:1", "1:3", "21:9", "9:21"}
SUPPORTED_RESOLUTIONS = {"1k", "2k", "4k"}


class GPTImageAPIError(RuntimeError):
    def __init__(self, message: str, data: Any = None, status: int | None = None):
        super().__init__(message)
        self.data = data
        self.status = status


@register(
    PLUGIN_NAME,
    "Codex",
    "使用 gpt-image-2 兼容接口的文生图与图生图插件。",
    "v1.0.0",
)
class GPTImage2Plugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.api_base_url = str(
            self.config.get("api_base_url", "https://api.apimart.ai/v1")
        ).rstrip("/")
        self.api_key = str(self.config.get("api_key", "")).strip()
        self.model = str(self.config.get("model", "gpt-image-2")).strip()
        if not self.model:
            self.model = "gpt-image-2"
        self.default_size = str(self.config.get("default_size", "1:1")).strip()
        self.default_resolution = str(
            self.config.get("default_resolution", "1k")
        ).strip()
        self.include_resolution = bool(self.config.get("include_resolution", True))
        self.official_fallback = bool(self.config.get("official_fallback", False))
        self.auto_wait = bool(self.config.get("auto_wait", True))
        self.initial_delay = int(self.config.get("initial_delay", 12))
        self.poll_interval = int(self.config.get("poll_interval", 5))
        self.poll_timeout = int(self.config.get("poll_timeout", 180))
        self.request_timeout = int(self.config.get("request_timeout", 60))
        self.max_reference_images = int(self.config.get("max_reference_images", 16))
        self.include_result_link = bool(self.config.get("include_result_link", True))
        self.session: aiohttp.ClientSession | None = None

        if not self.api_key:
            logger.warning(
                "[GPTImage2] API key is not configured. Register: %s",
                REGISTER_URL,
            )

    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=max(self.request_timeout, 30)),
                connector=aiohttp.TCPConnector(enable_cleanup_closed=True),
            )
        return self.session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _config_error(self) -> str | None:
        if not self.api_base_url.startswith(("http://", "https://")):
            return "API Base URL 配置无效，必须以 http:// 或 https:// 开头。"
        if not self.api_key:
            return f"请先在插件配置里填写 API Key。\n注册链接：👉 {REGISTER_URL}"
        return None

    @staticmethod
    def _join_api_url(base_url: str, path: str) -> str:
        base = base_url.rstrip("/")
        normalized_path = "/" + path.lstrip("/")
        if normalized_path == "/images/generations" and base.endswith(
            "/images/generations"
        ):
            return base
        if normalized_path.startswith("/tasks/"):
            for suffix in ("/images/generations",):
                if base.endswith(suffix):
                    return f"{base[: -len(suffix)]}{normalized_path}"
        return f"{base}{normalized_path}"

    @staticmethod
    def _format_api_error(data: Any, status: int | None = None) -> str:
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                code = error.get("code", status)
                message = error.get("message", "未知错误")
                error_type = error.get("type", "api_error")
                hint = GPTImage2Plugin._api_error_hint(code, message)
                return f"API 错误 {code} ({error_type}): {message}{hint}"
            code = data.get("code", status)
            message = data.get("message") or data.get("msg")
            if message:
                hint = GPTImage2Plugin._api_error_hint(code, message)
                return f"API 错误 {code}: {message}{hint}"
        return f"API 错误 {status}: {data}"

    @staticmethod
    def _api_error_hint(code: Any, message: Any) -> str:
        text = f"{code} {message}".lower()
        if "chat-requirements failed" in text or "chatgpt upstream 401" in text:
            return (
                "\n提示：这是接口站点转发到 ChatGPT 上游时的鉴权失败。"
                "插件会在可行时自动尝试 OpenAI 风格图像模型；如果仍失败，"
                "请检查该站点上游账号/Cookie 或模型通道。"
            )
        if "auth_required" in text:
            return "\n提示：上游要求鉴权，请检查接口站点、API Key 或当前模型通道。"
        return ""

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = await self._get_session()
        url = self._join_api_url(self.api_base_url, path)
        try:
            async with session.request(
                method,
                url,
                json=payload,
                headers=self._headers(),
            ) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text) if text else {}
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"HTTP {resp.status}: 返回不是 JSON：{text[:300]}"
                    ) from exc

                if resp.status >= 400:
                    raise GPTImageAPIError(
                        self._format_api_error(data, resp.status),
                        data,
                        resp.status,
                    )
                if isinstance(data, dict) and data.get("error"):
                    raise GPTImageAPIError(
                        self._format_api_error(data, resp.status),
                        data,
                        resp.status,
                    )
                return data
        except TimeoutError as exc:
            raise RuntimeError(f"请求超时（{self.request_timeout}s）。") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"网络请求失败：{exc}") from exc

    @staticmethod
    def _first_data_item(data: dict[str, Any]) -> dict[str, Any]:
        items = data.get("data")
        if isinstance(items, list) and items:
            first = items[0]
            return first if isinstance(first, dict) else {}
        if isinstance(items, dict):
            return items
        return {}

    @staticmethod
    def _split_values(value: str) -> list[str]:
        if not value:
            return []
        if value.startswith("data:image/"):
            return [value.strip()]
        return [item.strip() for item in re.split(r"[,，\s]+", value) if item.strip()]

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return value.lower() in {"1", "true", "yes", "y", "on", "是", "开", "开启"}

    def _parse_options(self, text: str) -> tuple[dict[str, str], str]:
        options: dict[str, str] = {}
        prompt_parts: list[str] = []
        tokens = text.split()
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("--"):
                key = token[2:].strip().replace("-", "_")
                if not key:
                    index += 1
                    continue
                if "=" in key:
                    key, value = key.split("=", 1)
                    options[key] = value.strip()
                elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                    options[key] = tokens[index + 1].strip()
                    index += 1
                else:
                    options[key] = "true"
            else:
                prompt_parts.append(token)
            index += 1
        return options, " ".join(prompt_parts).strip()

    @staticmethod
    def _is_pixel_size(size: str) -> bool:
        return bool(re.fullmatch(r"[1-9]\d{1,4}x[1-9]\d{1,4}", size))

    @staticmethod
    def _is_ratio_size(size: str) -> bool:
        return bool(re.fullmatch(r"[1-9]\d{0,2}:[1-9]\d{0,2}", size))

    @staticmethod
    def _is_openai_style_image_model(model: str) -> bool:
        normalized = model.strip().lower()
        return normalized in {"gpt-image-1", "dall-e-2", "dall-e-3"}

    @staticmethod
    def _model_supports_resolution(model: str) -> bool:
        return model.strip().lower().startswith("gpt-image-2")

    @staticmethod
    def _ratio_to_openai_size(size: str) -> str:
        if size == "auto" or GPTImage2Plugin._is_pixel_size(size):
            return size
        if not GPTImage2Plugin._is_ratio_size(size):
            return size
        width_ratio, height_ratio = (int(part) for part in size.split(":", 1))
        if width_ratio == height_ratio:
            return "1024x1024"
        if width_ratio > height_ratio:
            return "1536x1024"
        return "1024x1536"

    def _adapt_size_for_model(self, model: str, size: str) -> str:
        if self._is_openai_style_image_model(model):
            return self._ratio_to_openai_size(size)
        return size

    def _should_include_resolution(
        self,
        model: str,
        options: dict[str, str],
    ) -> bool:
        if "include_resolution" in options:
            return self._parse_bool(options["include_resolution"])
        if "no_resolution" in options:
            return not self._parse_bool(options["no_resolution"])
        return self.include_resolution and self._model_supports_resolution(model)

    def _validate_size_resolution(self, size: str, resolution: str) -> str | None:
        if size not in SUPPORTED_SIZES and not self._is_pixel_size(size):
            return (
                "size 不合法。支持 auto、1:1、3:2、2:3、4:3、3:4、5:4、4:5、"
                "16:9、9:16、2:1、1:2、3:1、1:3、21:9、9:21，或像素尺寸如 1881x836。"
            )
        if resolution not in SUPPORTED_RESOLUTIONS:
            return "resolution 不合法，仅支持 1k / 2k / 4k。"
        if (
            resolution == "4k"
            and size != "auto"
            and size in SUPPORTED_SIZES
            and size not in FOUR_K_SIZES
        ):
            return "4K 仅支持 16:9 / 9:16 / 2:1 / 1:2 / 3:1 / 1:3 / 21:9 / 9:21。"
        return None

    def _build_payload(
        self,
        prompt: str,
        options: dict[str, str],
        image_urls: list[str],
    ) -> dict[str, Any]:
        model = options.get("model", self.model).strip() or self.model
        size = options.get("size", self.default_size).strip()
        size = self._adapt_size_for_model(model, size)
        resolution = options.get("resolution", self.default_resolution).strip().lower()
        include_resolution = self._should_include_resolution(model, options)
        validation_error = self._validate_size_resolution(size, resolution)
        if validation_error:
            raise ValueError(validation_error)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
        }
        if include_resolution:
            payload["resolution"] = resolution
        if image_urls:
            payload["image_urls"] = image_urls
        if "official_fallback" in options:
            payload["official_fallback"] = self._parse_bool(
                options["official_fallback"]
            )
        elif self.official_fallback:
            payload["official_fallback"] = True
        return payload

    def _iter_image_components(self, chain: list[Any] | None):
        for comp in chain or []:
            if isinstance(comp, Comp.Reply):
                yield from self._iter_image_components(getattr(comp, "chain", None))
            elif isinstance(comp, Comp.Image):
                yield comp

    @staticmethod
    def _guess_mime_from_base64(data: str) -> str:
        try:
            raw = base64.b64decode(data[:128] + "===")
        except Exception:
            return "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
            return "image/gif"
        if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
            return "image/webp"
        return "image/png"

    def _base64_to_data_uri(self, data: str) -> str:
        clean_data = data.removeprefix("base64://")
        mime = self._guess_mime_from_base64(clean_data)
        return f"data:{mime};base64,{clean_data}"

    async def _image_component_to_api_value(self, comp: Comp.Image) -> str | None:
        value = (
            getattr(comp, "url", None) or getattr(comp, "file", None) or ""
        ).strip()
        if value.startswith(("http://", "https://", "data:image/")):
            return value
        if value.startswith("base64://"):
            return self._base64_to_data_uri(value)

        try:
            data = await comp.convert_to_base64()
        except Exception as exc:
            logger.warning("[GPTImage2] Failed to convert reference image: %s", exc)
            return None
        return self._base64_to_data_uri(data) if data else None

    async def _collect_reference_images(
        self,
        event: AstrMessageEvent,
        explicit_images: list[str],
    ) -> list[str]:
        images: list[str] = []
        for image in explicit_images:
            if image.startswith(("http://", "https://", "data:image/")):
                images.append(image)
            elif image.startswith("base64://"):
                images.append(self._base64_to_data_uri(image))

        for comp in self._iter_image_components(event.message_obj.message):
            image = await self._image_component_to_api_value(comp)
            if image:
                images.append(image)

        deduped = list(dict.fromkeys(images))
        limit = max(1, min(self.max_reference_images, 16))
        return deduped[:limit]

    @staticmethod
    def _normalize_image_ref(
        value: Any,
        *,
        allow_raw_base64: bool = False,
    ) -> str | None:
        if not isinstance(value, str):
            return None
        image = value.strip()
        if image.startswith(("http://", "https://")):
            return image
        if image.startswith("base64://"):
            base64_data = image.removeprefix("base64://").strip()
            return f"base64://{base64_data}" if base64_data else None
        if image.startswith("data:image/"):
            match = re.match(r"^data:image/[^;]+;base64,(.+)$", image, re.S)
            if not match:
                return None
            base64_data = match.group(1).strip()
            return f"base64://{base64_data}" if base64_data else None
        if allow_raw_base64 and image:
            return f"base64://{image}"
        return None

    @staticmethod
    def _extract_image_refs(value: Any) -> list[str]:
        refs: list[str] = []

        def append(item: Any, *, allow_raw_base64: bool = False) -> None:
            if isinstance(item, list):
                for sub_item in item:
                    append(sub_item, allow_raw_base64=allow_raw_base64)
                return
            ref = GPTImage2Plugin._normalize_image_ref(
                item,
                allow_raw_base64=allow_raw_base64,
            )
            if ref:
                refs.append(ref)

        if isinstance(value, dict):
            result = value.get("result")
            if isinstance(result, dict):
                images = result.get("images")
                if isinstance(images, list):
                    for image in images:
                        if isinstance(image, dict):
                            append(image.get("url"))
                            append(image.get("image_url"))
                            append(image.get("download_url"))
                            append(image.get("b64_json"), allow_raw_base64=True)
                        else:
                            append(image)

            for key in ("url", "image_url", "download_url"):
                append(value.get(key))
            for key in ("b64_json", "image_base64", "base64"):
                append(value.get(key), allow_raw_base64=True)

            for item in value.values():
                refs.extend(GPTImage2Plugin._extract_image_refs(item))
        elif isinstance(value, list):
            for item in value:
                refs.extend(GPTImage2Plugin._extract_image_refs(item))
        return list(dict.fromkeys(refs))

    @staticmethod
    def _extract_image_urls(value: Any) -> list[str]:
        return [
            ref
            for ref in GPTImage2Plugin._extract_image_refs(value)
            if ref.startswith(("http://", "https://"))
        ]

    @staticmethod
    def _compact_response_for_error(data: Any) -> Any:
        if isinstance(data, dict):
            compact: dict[str, Any] = {}
            for key, value in data.items():
                if key in {"b64_json", "image_base64", "base64"} and isinstance(
                    value,
                    str,
                ):
                    compact[key] = f"<base64:{len(value)} chars>"
                else:
                    compact[key] = GPTImage2Plugin._compact_response_for_error(value)
            return compact
        if isinstance(data, list):
            return [GPTImage2Plugin._compact_response_for_error(item) for item in data]
        if isinstance(data, str) and len(data) > 500:
            return f"{data[:200]}...<truncated:{len(data)} chars>"
        return data

    @staticmethod
    def _api_exception_text(exc: Exception) -> str:
        parts = [str(exc)]
        data = getattr(exc, "data", None)
        if data is not None:
            try:
                parts.append(json.dumps(data, ensure_ascii=False))
            except TypeError:
                parts.append(str(data))
        return " ".join(parts).lower()

    @staticmethod
    def _is_resolution_param_error(exc: Exception) -> bool:
        text = GPTImage2Plugin._api_exception_text(exc)
        if "resolution" not in text:
            return False
        markers = {
            "unknown",
            "unsupported",
            "unrecognized",
            "invalid",
            "not support",
            "not supported",
            "not allowed",
            "not permitted",
            "extra_forbidden",
        }
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_size_param_error(exc: Exception) -> bool:
        text = GPTImage2Plugin._api_exception_text(exc)
        if "size" not in text:
            return False
        markers = {"invalid", "unsupported", "not support", "not supported"}
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_chatgpt_upstream_auth_error(exc: Exception) -> bool:
        text = GPTImage2Plugin._api_exception_text(exc)
        return "chat-requirements failed" in text or "chatgpt upstream 401" in text

    @staticmethod
    def _payload_signature(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    def _retry_payload_for_error(
        self,
        payload: dict[str, Any],
        exc: Exception,
        *,
        allow_model_fallback: bool,
    ) -> tuple[dict[str, Any], str] | None:
        if "resolution" in payload and self._is_resolution_param_error(exc):
            retry_payload = dict(payload)
            retry_payload.pop("resolution", None)
            return retry_payload, "接口不支持 resolution 字段，已自动去掉后重试"

        model = str(payload.get("model", "")).strip().lower()
        if (
            allow_model_fallback
            and model == "gpt-image-2"
            and self._is_chatgpt_upstream_auth_error(exc)
        ):
            retry_payload = dict(payload)
            retry_payload["model"] = "gpt-image-1"
            retry_payload.pop("resolution", None)
            retry_payload["size"] = self._ratio_to_openai_size(
                str(retry_payload.get("size", "auto")).strip()
            )
            return (
                retry_payload,
                "gpt-image-2 上游鉴权失败，已自动尝试 gpt-image-1 兼容参数",
            )

        if self._is_size_param_error(exc):
            size = str(payload.get("size", "")).strip()
            adapted_size = self._ratio_to_openai_size(size)
            if adapted_size != size:
                retry_payload = dict(payload)
                retry_payload["size"] = adapted_size
                retry_payload.pop("resolution", None)
                return retry_payload, "接口不接受比例 size，已自动改为像素尺寸后重试"

        return None

    async def _submit_generation(
        self,
        payload: dict[str, Any],
        *,
        allow_model_fallback: bool = True,
    ) -> tuple[str, list[str]]:
        attempted = {self._payload_signature(payload)}
        while True:
            try:
                return await self._submit_generation_once(payload)
            except GPTImageAPIError as exc:
                retry = self._retry_payload_for_error(
                    payload,
                    exc,
                    allow_model_fallback=allow_model_fallback,
                )
                if retry is None:
                    raise
                retry_payload, reason = retry
                signature = self._payload_signature(retry_payload)
                if signature in attempted:
                    raise
                attempted.add(signature)
                logger.warning("[GPTImage2] %s", reason)
                payload = retry_payload

    async def _submit_generation_once(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, list[str]]:
        data = await self._request("POST", "/images/generations", payload)
        item = self._first_data_item(data)
        task_id = str(
            item.get("task_id")
            or item.get("task")
            or item.get("id")
            or data.get("task_id")
            or data.get("task")
            or data.get("id")
            or ""
        ).strip()
        image_refs = self._extract_image_refs(data)
        if not task_id and not image_refs:
            compact = self._compact_response_for_error(data)
            raise RuntimeError(
                "提交成功但未找到 task_id 或图片数据："
                f"{json.dumps(compact, ensure_ascii=False)[:1500]}"
            )
        return task_id, image_refs

    async def _get_task(self, task_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/tasks/{task_id}")

    async def _poll_task(self, task_id: str) -> dict[str, Any]:
        await asyncio.sleep(max(self.initial_delay, 0))
        deadline = asyncio.get_running_loop().time() + self.poll_timeout
        last_data: dict[str, Any] = {}
        while asyncio.get_running_loop().time() < deadline:
            data = await self._get_task(task_id)
            last_data = data
            task = self._first_data_item(data)
            status = str(task.get("status", "")).lower()
            if status in {"completed", "succeeded", "success"}:
                return data
            if status in {"failed", "cancelled", "canceled"}:
                error = task.get("error") or data.get("error") or "任务失败"
                raise RuntimeError(f"任务 {task_id} 失败：{error}")
            await asyncio.sleep(max(self.poll_interval, 3))
        raise RuntimeError(f"任务 {task_id} 查询超时，最后响应：{last_data}")

    def _format_task_status(self, data: dict[str, Any]) -> tuple[str, list[str]]:
        task = self._first_data_item(data)
        if not task:
            image_refs = self._extract_image_refs(data)
            if image_refs:
                return "已解析到图片数据。", image_refs
            return f"任务返回：\n{json.dumps(data, ensure_ascii=False, indent=2)}", []

        task_id = task.get("id") or task.get("task_id") or ""
        status = task.get("status", "unknown")
        progress = task.get("progress")
        cost = task.get("cost")
        error = task.get("error") or data.get("error")
        image_refs = self._extract_image_refs(data)
        urls = [ref for ref in image_refs if ref.startswith(("http://", "https://"))]

        lines = [f"任务状态：{status}"]
        if task_id:
            lines.append(f"任务 ID：{task_id}")
        if progress is not None:
            lines.append(f"进度：{progress}%")
        if cost is not None:
            lines.append(f"费用：{cost}")
        if urls:
            lines.append("图片链接：")
            lines.extend(urls[:4])
        elif image_refs:
            lines.append("已解析到图片数据。")
        if error:
            lines.append(f"错误：{error}")
        return "\n".join(lines), image_refs

    def _result_chain(self, image_refs: list[str]) -> list[Any]:
        chain: list[Any] = []
        urls: list[str] = []
        for image_ref in image_refs[:4]:
            if image_ref.startswith(("http://", "https://")):
                urls.append(image_ref)
                chain.append(Comp.Image.fromURL(image_ref))
            elif image_ref.startswith("base64://"):
                chain.append(
                    Comp.Image.fromBase64(image_ref.removeprefix("base64://"))
                )

        if self.include_result_link and urls:
            links = "\n".join(urls[:4])
            chain.append(Comp.Plain(f"\n图片生成完成\n{links}"))
        else:
            chain.append(Comp.Plain("\n图片生成完成"))
        return chain

    async def _handle_generate(self, event: AstrMessageEvent, args: str):
        error = self._config_error()
        if error:
            yield event.plain_result(error)
            return

        options, prompt = self._parse_options(args)
        if not prompt:
            yield event.plain_result(self._help_text(short=True))
            return

        explicit_images = self._split_values(options.get("images", ""))
        image_urls = await self._collect_reference_images(event, explicit_images)
        try:
            payload = self._build_payload(prompt, options, image_urls)
        except ValueError as exc:
            yield event.plain_result(f"参数错误：{exc}")
            return

        wait = self.auto_wait
        if "wait" in options:
            wait = self._parse_bool(options["wait"])
        if "no_wait" in options:
            wait = not self._parse_bool(options["no_wait"])

        yield event.plain_result("已提交图片生成请求，请稍候。")
        try:
            task_id, image_refs = await self._submit_generation(
                payload,
                allow_model_fallback="model" not in options,
            )
            if image_refs:
                yield event.chain_result(self._result_chain(image_refs))
                return
            if not wait:
                yield event.plain_result(
                    f"任务已提交：{task_id}\n查询：/gptimage 查询 {task_id}"
                )
                return

            data = await self._poll_task(task_id)
            _, image_refs = self._format_task_status(data)
            if not image_refs:
                yield event.plain_result(
                    f"任务已完成，但未解析到图片数据：\n{json.dumps(data, ensure_ascii=False)[:1500]}"
                )
                return
            yield event.chain_result(self._result_chain(image_refs))
        except Exception as exc:
            logger.error("[GPTImage2] Image generation failed: %s", exc, exc_info=True)
            yield event.plain_result(f"图片生成失败：{exc}")

    @filter.command_group("gptimage", alias={"gpt图"})
    def gptimage(self):
        pass

    @gptimage.command("生成", alias={"gen", "create", "画图"})
    async def generate_group(self, event: AstrMessageEvent):
        """使用 gpt-image-2 生成图片。"""
        parts = event.message_str.split(maxsplit=2)
        args = parts[2] if len(parts) >= 3 else ""
        async for result in self._handle_generate(event, args):
            yield result

    @filter.command("gpt生图", alias={"gpt画图", "生图2"})
    async def generate_shortcut(self, event: AstrMessageEvent):
        """快捷使用 gpt-image-2 生成图片。"""
        parts = event.message_str.split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        async for result in self._handle_generate(event, args):
            yield result

    @gptimage.command("查询", alias={"task", "status"})
    async def query_task(self, event: AstrMessageEvent, task_id: str):
        """查询 gpt-image-2 异步任务状态。"""
        error = self._config_error()
        if error:
            yield event.plain_result(error)
            return
        try:
            data = await self._get_task(task_id)
            text, urls = self._format_task_status(data)
            if urls and str(self._first_data_item(data).get("status", "")).lower() in {
                "completed",
                "succeeded",
                "success",
            }:
                yield event.chain_result(self._result_chain(urls))
            else:
                yield event.plain_result(text)
        except Exception as exc:
            logger.error("[GPTImage2] Query task failed: %s", exc, exc_info=True)
            yield event.plain_result(f"查询失败：{exc}")

    @gptimage.command("帮助", alias={"help"})
    async def help_command(self, event: AstrMessageEvent):
        """显示 gpt-image-2 生图插件帮助。"""
        yield event.plain_result(self._help_text())

    def _help_text(self, *, short: bool = False) -> str:
        text = """gpt-image-2 生图插件

用法：
/gpt生图 一只橘猫坐在窗台上看夕阳，水彩画风格 --size 16:9 --resolution 2k
/gptimage 生成 星空下的古老城堡 --size 16:9 --resolution 4k
/gptimage 查询 task_xxx

图生图：
直接发送或引用图片，并附带 /gpt生图 提示词；也可用 --images 传公网 URL，多个用逗号分隔。

常用参数：
--model gpt-image-2
--size 1:1|16:9|9:16|auto
--resolution 1k|2k|4k
--no-resolution true|false
--wait true|false
--official-fallback true|false

注册链接：👉 https://apimart.ai/register?aff=J3ZjCO"""
        if short:
            return "\n".join(text.splitlines()[:8])
        return text
