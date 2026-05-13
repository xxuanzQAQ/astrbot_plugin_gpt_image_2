import asyncio
import base64
import json
import re
from typing import Any
from urllib.parse import urlsplit

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.core.utils.quoted_message.image_resolver import ImageResolver
    from astrbot.core.utils.quoted_message_parser import extract_quoted_message_images
except Exception:  # pragma: no cover - keep compatibility with older AstrBot builds
    ImageResolver = None
    extract_quoted_message_images = None

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
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504, 522, 524}


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
        self.transient_retries = int(self.config.get("transient_retries", 2))
        self.transient_retry_delay = int(self.config.get("transient_retry_delay", 5))
        self.max_reference_images = int(self.config.get("max_reference_images", 16))
        self.include_result_link = bool(self.config.get("include_result_link", True))
        self.debug_log_payload = bool(self.config.get("debug_log_payload", False))
        self.image_to_image_endpoint = str(
            self.config.get("image_to_image_endpoint", "auto")
        ).strip()
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

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _summarize_image_value_for_log(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        image = value.strip()
        lower_image = image.lower()
        if lower_image.startswith("data:image/"):
            header, sep, payload = image.partition(",")
            if sep:
                return f"{header},<base64:{len(payload)} chars>"
            return f"<data-image:{len(image)} chars>"
        if lower_image.startswith("base64://"):
            return f"base64://<base64:{len(image.removeprefix('base64://'))} chars>"
        if len(image) > 500:
            return f"{image[:200]}...<truncated:{len(image)} chars>"
        return image

    @classmethod
    def _sanitize_for_log(cls, value: Any, *, key: str = "") -> Any:
        key_lower = key.lower()
        if key_lower in {
            "authorization",
            "api_key",
            "apikey",
            "token",
            "image",
            "image[]",
            "mask",
        }:
            return "<redacted>"
        if key_lower in {"image_urls", "images"} and isinstance(value, list):
            return [cls._summarize_image_value_for_log(item) for item in value]
        if key_lower in {"b64_json", "image_base64", "base64"} and isinstance(
            value,
            str,
        ):
            return f"<base64:{len(value)} chars>"
        if isinstance(value, dict):
            return {
                str(item_key): cls._sanitize_for_log(item_value, key=str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [cls._sanitize_for_log(item) for item in value]
        if isinstance(value, str):
            return cls._summarize_image_value_for_log(value)
        return value

    def _log_request_payload(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
    ) -> None:
        if not isinstance(payload, dict):
            logger.info("[GPTImage2] 请求：%s %s", method, url)
            return

        image_urls = payload.get("image_urls")
        image_count = len(image_urls) if isinstance(image_urls, list) else 0
        prompt = str(payload.get("prompt", ""))
        logger.info(
            "[GPTImage2] 请求：%s %s model=%s size=%s resolution=%s "
            "image_urls=%s prompt_len=%s",
            method,
            url,
            payload.get("model"),
            payload.get("size"),
            payload.get("resolution", "<omitted>"),
            image_count,
            len(prompt),
        )
        if image_count:
            logger.info(
                "[GPTImage2] 请求参考图：%s",
                json.dumps(
                    self._sanitize_for_log(image_urls, key="image_urls"),
                    ensure_ascii=False,
                ),
            )
        if self.debug_log_payload:
            logger.info(
                "[GPTImage2] 请求 headers：%s",
                json.dumps(
                    self._sanitize_for_log(self._headers()),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            logger.info(
                "[GPTImage2] 请求 payload：%s",
                json.dumps(
                    self._sanitize_for_log(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

    def _log_response_payload(self, status: int, data: Any) -> None:
        if not self.debug_log_payload:
            return
        logger.info(
            "[GPTImage2] 响应 HTTP %s：%s",
            status,
            json.dumps(
                self._sanitize_for_log(data),
                ensure_ascii=False,
                sort_keys=True,
            )[:3000],
        )

    def _log_multipart_request(
        self,
        method: str,
        url: str,
        fields: dict[str, Any],
        image_uploads: list[tuple[bytes, str, str]],
    ) -> None:
        logger.info(
            "[GPTImage2] 请求：%s %s multipart model=%s size=%s image=%s prompt_len=%s",
            method,
            url,
            fields.get("model"),
            fields.get("size"),
            len(image_uploads),
            len(str(fields.get("prompt", ""))),
        )
        logger.info(
            "[GPTImage2] 请求上传图片：%s",
            json.dumps(
                [
                    {
                        "filename": filename,
                        "content_type": content_type,
                        "bytes": len(data),
                    }
                    for data, filename, content_type in image_uploads
                ],
                ensure_ascii=False,
            ),
        )
        if self.debug_log_payload:
            logger.info(
                "[GPTImage2] 请求 headers：%s",
                json.dumps(
                    self._sanitize_for_log(self._auth_headers()),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            logger.info(
                "[GPTImage2] 请求 form fields：%s",
                json.dumps(
                    self._sanitize_for_log(fields),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

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
        image_endpoints = {"/images/generations", "/images/edits"}
        if normalized_path in image_endpoints:
            for suffix in image_endpoints:
                if base.endswith(suffix):
                    return f"{base[: -len(suffix)]}{normalized_path}"
        if normalized_path.startswith("/tasks/"):
            for suffix in image_endpoints:
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
        if (
            str(code) in {"502", "503", "504", "522", "524"}
            or "context deadline exceeded" in text
            or "gateway time" in text
        ):
            return (
                "\n提示：这是接口站点或上游模型服务超时/网关错误，"
                "插件会自动短间隔重试；如果多次失败，需要稍后再试或更换可用通道。"
            )
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
        self._log_request_payload(method, url, payload)
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
                    if resp.status >= 400:
                        raise GPTImageAPIError(
                            self._format_non_json_http_error(text, resp.status),
                            text,
                            resp.status,
                        ) from exc
                    raise RuntimeError(
                        f"HTTP {resp.status}: 返回不是 JSON：{text[:300]}"
                    ) from exc

                if resp.status >= 400:
                    self._log_response_payload(resp.status, data)
                    raise GPTImageAPIError(
                        self._format_api_error(data, resp.status),
                        data,
                        resp.status,
                    )
                if isinstance(data, dict) and data.get("error"):
                    self._log_response_payload(resp.status, data)
                    raise GPTImageAPIError(
                        self._format_api_error(data, resp.status),
                        data,
                        resp.status,
                    )
                self._log_response_payload(resp.status, data)
                return data
        except TimeoutError as exc:
            raise RuntimeError(f"请求超时（{self.request_timeout}s）。") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"网络请求失败：{exc}") from exc

    async def _request_multipart(
        self,
        method: str,
        path: str,
        fields: dict[str, Any],
        image_uploads: list[tuple[bytes, str, str]],
    ) -> dict[str, Any]:
        session = await self._get_session()
        url = self._join_api_url(self.api_base_url, path)
        self._log_multipart_request(method, url, fields, image_uploads)

        form = aiohttp.FormData()
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            form.add_field(key, str(value))
        for data, filename, content_type in image_uploads:
            form.add_field(
                "image",
                data,
                filename=filename,
                content_type=content_type,
            )

        try:
            async with session.request(
                method,
                url,
                data=form,
                headers=self._auth_headers(),
            ) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text) if text else {}
                except json.JSONDecodeError as exc:
                    if resp.status >= 400:
                        raise GPTImageAPIError(
                            self._format_non_json_http_error(text, resp.status),
                            text,
                            resp.status,
                        ) from exc
                    raise RuntimeError(
                        f"HTTP {resp.status}: 返回不是 JSON：{text[:300]}"
                    ) from exc

                if resp.status >= 400:
                    self._log_response_payload(resp.status, data)
                    raise GPTImageAPIError(
                        self._format_api_error(data, resp.status),
                        data,
                        resp.status,
                    )
                if isinstance(data, dict) and data.get("error"):
                    self._log_response_payload(resp.status, data)
                    raise GPTImageAPIError(
                        self._format_api_error(data, resp.status),
                        data,
                        resp.status,
                    )
                self._log_response_payload(resp.status, data)
                return data
        except TimeoutError as exc:
            raise RuntimeError(f"请求超时（{self.request_timeout}s）。") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"网络请求失败：{exc}") from exc

    @staticmethod
    def _format_non_json_http_error(text: str, status: int) -> str:
        body = re.sub(r"\s+", " ", text).strip()
        if "<html" in body.lower():
            if "504" in body or status == 504:
                body = "网关超时"
            elif "502" in body or status == 502:
                body = "上游网关错误"
            elif "503" in body or status == 503:
                body = "服务暂不可用"
            else:
                body = body[:300]
        else:
            body = body[:300]
        hint = GPTImage2Plugin._api_error_hint(status, body)
        return f"HTTP {status}: {body}{hint}"

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

    @staticmethod
    def _is_apimart_base_url(base_url: str) -> bool:
        try:
            host = urlsplit(base_url).netloc.lower()
        except Exception:
            return False
        return host.endswith("apimart.ai")

    @staticmethod
    def _normalize_endpoint_path(value: str) -> str:
        endpoint = value.strip()
        if not endpoint or endpoint.lower() == "auto":
            return "auto"
        if endpoint.startswith(("http://", "https://")):
            parsed = urlsplit(endpoint)
            endpoint = parsed.path
        endpoint = "/" + endpoint.lstrip("/")
        if endpoint.startswith("/v1/"):
            endpoint = endpoint[3:]
        if endpoint in {"/generations", "/image/generations"}:
            return "/images/generations"
        if endpoint in {"/edits", "/image/edits"}:
            return "/images/edits"
        return endpoint

    def _image_to_image_path(self) -> str:
        endpoint = self._normalize_endpoint_path(self.image_to_image_endpoint)
        if endpoint != "auto":
            return endpoint
        if self._is_apimart_base_url(self.api_base_url):
            return "/images/generations"
        return "/images/edits"

    def _edit_form_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        fields = {
            key: value
            for key, value in payload.items()
            if key not in {"image_urls", "resolution", "official_fallback"}
        }
        fields["size"] = self._ratio_to_openai_size(str(fields.get("size", "auto")))
        return fields

    def _iter_image_components(
        self,
        chain: list[Any] | None,
        *,
        include_reply_chain: bool = True,
    ):
        for comp in chain or []:
            if isinstance(comp, Comp.Reply):
                if include_reply_chain:
                    yield from self._iter_image_components(
                        getattr(comp, "chain", None),
                        include_reply_chain=include_reply_chain,
                    )
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

    @staticmethod
    def _image_extension_from_mime(mime: str) -> str:
        return {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
        }.get(mime.lower(), "png")

    @staticmethod
    def _parse_data_uri_image(value: str) -> tuple[bytes, str] | None:
        match = re.match(r"^data:(image/[^;]+);base64,(.+)$", value.strip(), re.S)
        if not match:
            return None
        mime = match.group(1).lower()
        try:
            data = base64.b64decode(match.group(2), validate=False)
        except Exception:
            return None
        return data, mime

    async def _download_image_upload(self, url: str) -> tuple[bytes, str] | None:
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                data = await resp.read()
                if resp.status >= 400:
                    logger.warning(
                        "[GPTImage2] 下载参考图失败：HTTP %s %s",
                        resp.status,
                        url[:200],
                    )
                    return None
                content_type = resp.headers.get("Content-Type", "").split(";", 1)[0]
        except Exception as exc:
            logger.warning("[GPTImage2] 下载参考图失败：%s (%s)", url[:200], exc)
            return None

        if not content_type.startswith("image/"):
            content_type = self._guess_mime_from_base64(base64.b64encode(data).decode())
        return data, content_type

    async def _image_value_to_upload(
        self,
        value: str,
        index: int,
    ) -> tuple[bytes, str, str] | None:
        image = value.strip()
        parsed = self._parse_data_uri_image(image)
        if parsed:
            data, content_type = parsed
        elif image.startswith("base64://"):
            base64_data = image.removeprefix("base64://")
            try:
                data = base64.b64decode(base64_data, validate=False)
            except Exception:
                return None
            content_type = self._guess_mime_from_base64(base64_data)
        elif image.startswith(("http://", "https://")):
            downloaded = await self._download_image_upload(image)
            if not downloaded:
                return None
            data, content_type = downloaded
        else:
            data_uri = await self._value_to_data_uri(image)
            if not data_uri:
                return None
            parsed = self._parse_data_uri_image(data_uri)
            if not parsed:
                return None
            data, content_type = parsed

        if not data:
            return None
        ext = self._image_extension_from_mime(content_type)
        return data, f"reference_{index}.{ext}", content_type

    async def _build_edit_uploads(
        self,
        image_urls: list[str],
    ) -> list[tuple[bytes, str, str]]:
        uploads: list[tuple[bytes, str, str]] = []
        for index, image in enumerate(image_urls, 1):
            upload = await self._image_value_to_upload(image, index)
            if upload:
                uploads.append(upload)
            else:
                logger.warning(
                    "[GPTImage2] 参考图无法转为 /images/edits 上传文件：%s",
                    self._summarize_image_value_for_log(image),
                )
        return uploads

    async def _value_to_data_uri(self, value: str) -> str | None:
        image = value.strip()
        if not image:
            return None
        if image.startswith("data:image/"):
            return image
        if image.startswith("base64://"):
            return self._base64_to_data_uri(image)
        try:
            data = await Comp.Image(file=image).convert_to_base64()
        except Exception:
            return None
        return self._base64_to_data_uri(data) if data else None

    async def _reference_value_to_api_value(
        self,
        event: AstrMessageEvent,
        value: Any,
        *,
        prefer_base64: bool,
        resolve_remote: bool = True,
    ) -> str | None:
        if not isinstance(value, str):
            return None
        image = value.strip()
        if not image:
            return None

        if image.startswith("data:image/"):
            return image
        if image.startswith("base64://"):
            return self._base64_to_data_uri(image)

        if prefer_base64:
            data_uri = await self._value_to_data_uri(image)
            if data_uri:
                return data_uri

        if image.startswith(("http://", "https://")):
            return image

        if not prefer_base64:
            data_uri = await self._value_to_data_uri(image)
            if data_uri:
                return data_uri

        if resolve_remote and ImageResolver is not None:
            try:
                resolved_refs = await ImageResolver(event).resolve_for_llm([image])
            except Exception as exc:
                logger.warning(
                    "[GPTImage2] 解析参考图失败：%s (%s)",
                    self._summarize_image_value_for_log(image),
                    exc,
                )
                resolved_refs = []
            for resolved_ref in resolved_refs:
                resolved = await self._reference_value_to_api_value(
                    event,
                    resolved_ref,
                    prefer_base64=prefer_base64,
                    resolve_remote=False,
                )
                if resolved:
                    return resolved

        return None

    async def _image_component_to_api_value(
        self,
        event: AstrMessageEvent,
        comp: Comp.Image,
    ) -> str | None:
        candidates: list[str] = []
        for attr in ("url", "file", "path"):
            value = getattr(comp, attr, None)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value and value not in candidates:
                candidates.append(value)

        for candidate in candidates:
            image = await self._reference_value_to_api_value(
                event,
                candidate,
                prefer_base64=True,
            )
            if image:
                return image

        try:
            data = await comp.convert_to_base64()
        except Exception as exc:
            logger.warning("[GPTImage2] 转换消息参考图失败：%s", exc)
            return None
        return self._base64_to_data_uri(data) if data else None

    async def _collect_quoted_reference_images(
        self,
        event: AstrMessageEvent,
    ) -> list[str]:
        if extract_quoted_message_images is None:
            return []

        reply_components = [
            comp
            for comp in getattr(event.message_obj, "message", []) or []
            if isinstance(comp, Comp.Reply)
        ]
        images: list[str] = []
        for reply in reply_components:
            try:
                quoted_refs = await extract_quoted_message_images(event, reply)
            except Exception as exc:
                logger.warning(
                    "[GPTImage2] 提取引用图片失败：reply_id=%s, error=%s",
                    getattr(reply, "id", None),
                    exc,
                    exc_info=True,
                )
                continue

            for ref in quoted_refs:
                image = await self._reference_value_to_api_value(
                    event,
                    ref,
                    prefer_base64=True,
                )
                if image:
                    images.append(image)
                else:
                    logger.warning(
                        "[GPTImage2] 引用图片无法转换为接口可用格式：%s",
                        self._summarize_image_value_for_log(ref),
                    )
        return images

    async def _collect_reference_images(
        self,
        event: AstrMessageEvent,
        explicit_images: list[str],
    ) -> list[str]:
        images: list[str] = []
        explicit_count = 0
        message_count = 0
        quoted_count = 0
        for image in explicit_images:
            api_value = await self._reference_value_to_api_value(
                event,
                image,
                prefer_base64=False,
            )
            if api_value:
                images.append(api_value)
                explicit_count += 1
            else:
                logger.warning(
                    "[GPTImage2] --images 参考图无法转换为接口可用格式：%s",
                    self._summarize_image_value_for_log(image),
                )

        for comp in self._iter_image_components(
            event.message_obj.message,
            include_reply_chain=extract_quoted_message_images is None,
        ):
            image = await self._image_component_to_api_value(event, comp)
            if image:
                images.append(image)
                message_count += 1

        quoted_images = await self._collect_quoted_reference_images(event)
        images.extend(quoted_images)
        quoted_count = len(quoted_images)
        deduped = list(dict.fromkeys(images))
        limit = max(1, min(self.max_reference_images, 16))
        limited = deduped[:limit]
        logger.info(
            "[GPTImage2] 参考图收集：explicit=%s message=%s quoted=%s total=%s/%s",
            explicit_count,
            message_count,
            quoted_count,
            len(limited),
            len(deduped),
        )
        if not limited:
            logger.info("[GPTImage2] 未收集到参考图，本次会按文生图提交。")
        elif self.debug_log_payload:
            logger.info(
                "[GPTImage2] 参考图提交值：%s",
                json.dumps(
                    self._sanitize_for_log(limited, key="image_urls"),
                    ensure_ascii=False,
                ),
            )
        return limited

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
    def _is_transient_upstream_error(exc: Exception) -> bool:
        status = getattr(exc, "status", None)
        if isinstance(status, int) and status in TRANSIENT_STATUS_CODES:
            return True
        text = GPTImage2Plugin._api_exception_text(exc)
        markers = {
            "bad gateway",
            "gateway timeout",
            "gateway time-out",
            "upstream timeout",
            "upstream timed out",
            "context deadline exceeded",
            "context canceled",
            "poll error",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "timeout awaiting",
        }
        return any(marker in text for marker in markers)

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
        transient_attempts = 0
        while True:
            try:
                return await self._submit_generation_once(payload)
            except GPTImageAPIError as exc:
                if (
                    self._is_transient_upstream_error(exc)
                    and transient_attempts < max(self.transient_retries, 0)
                ):
                    transient_attempts += 1
                    delay = max(self.transient_retry_delay, 1)
                    logger.warning(
                        "[GPTImage2] 上游临时错误，%ss 后自动重试第 %s/%s 次：%s",
                        delay,
                        transient_attempts,
                        max(self.transient_retries, 0),
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue

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
        image_urls = payload.get("image_urls")
        endpoint = (
            self._image_to_image_path()
            if isinstance(image_urls, list) and image_urls
            else "/images/generations"
        )
        if endpoint == "/images/edits":
            uploads = await self._build_edit_uploads(image_urls)
            if not uploads:
                raise RuntimeError("图生图失败：参考图无法转成 /images/edits 上传文件。")
            data = await self._request_multipart(
                "POST",
                endpoint,
                self._edit_form_fields(payload),
                uploads,
            )
        else:
            data = await self._request("POST", endpoint, payload)
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
            try:
                data = await self._get_task(task_id)
            except GPTImageAPIError as exc:
                if self._is_transient_upstream_error(exc):
                    last_data = {"error": str(exc)}
                    logger.warning(
                        "[GPTImage2] 查询任务遇到上游临时错误，将继续轮询：%s",
                        exc,
                    )
                    await asyncio.sleep(max(self.poll_interval, 3))
                    continue
                raise
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
