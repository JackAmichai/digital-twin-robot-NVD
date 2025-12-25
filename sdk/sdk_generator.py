"""SDK code generator from OpenAPI specifications."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import json
import yaml


class Language(Enum):
    """Supported SDK languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    CSHARP = "csharp"


@dataclass
class SDKConfig:
    """SDK generation configuration."""
    
    language: Language = Language.PYTHON
    package_name: str = "robotics_sdk"
    version: str = "1.0.0"
    output_dir: Path = field(default_factory=lambda: Path("./generated_sdk"))
    base_url: str = "http://localhost:8000"
    include_models: bool = True
    include_async: bool = True
    author: str = ""
    license: str = "MIT"


class SDKGenerator:
    """Generate client SDKs from OpenAPI spec."""
    
    def __init__(self, openapi_path: Path | str):
        self.openapi_path = Path(openapi_path)
        self.spec = self._load_spec()
    
    def _load_spec(self) -> dict[str, Any]:
        """Load OpenAPI specification."""
        with open(self.openapi_path) as f:
            if self.openapi_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            return json.load(f)
    
    def generate(self, config: SDKConfig) -> Path:
        """Generate SDK for specified language."""
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        if config.language == Language.PYTHON:
            return self._generate_python(config)
        elif config.language == Language.TYPESCRIPT:
            return self._generate_typescript(config)
        elif config.language == Language.GO:
            return self._generate_go(config)
        else:
            raise ValueError(f"Unsupported language: {config.language}")
    
    def _generate_python(self, config: SDKConfig) -> Path:
        """Generate Python SDK."""
        pkg_dir = config.output_dir / config.package_name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate __init__.py
        init_content = self._python_init(config)
        (pkg_dir / "__init__.py").write_text(init_content)
        
        # Generate client.py
        client_content = self._python_client(config)
        (pkg_dir / "client.py").write_text(client_content)
        
        # Generate models.py
        if config.include_models:
            models_content = self._python_models(config)
            (pkg_dir / "models.py").write_text(models_content)
        
        # Generate setup.py
        setup_content = self._python_setup(config)
        (config.output_dir / "setup.py").write_text(setup_content)
        
        return config.output_dir
    
    def _python_init(self, config: SDKConfig) -> str:
        """Generate Python __init__.py."""
        return f'''"""
{config.package_name} - Auto-generated SDK
Version: {config.version}
"""

from {config.package_name}.client import RoboticsClient

__version__ = "{config.version}"
__all__ = ["RoboticsClient"]
'''
    
    def _python_client(self, config: SDKConfig) -> str:
        """Generate Python client."""
        methods = self._extract_methods()
        method_code = "\n".join(
            self._python_method(m, config.include_async)
            for m in methods
        )
        
        return f'''"""Robotics API Client."""

import httpx
from typing import Any, Optional
{"import asyncio" if config.include_async else ""}

class RoboticsClient:
    """Client for Digital Twin Robotics API."""
    
    def __init__(
        self,
        base_url: str = "{config.base_url}",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {{"Content-Type": "application/json"}}
        if self.api_key:
            headers["Authorization"] = f"Bearer {{self.api_key}}"
        return headers
    
    @property
    def client(self) -> httpx.Client:
        """Get sync HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._client
    
    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._async_client
    
    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
    
    async def aclose(self) -> None:
        """Close async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
    
{method_code}
'''
    
    def _python_method(self, method: dict, include_async: bool) -> str:
        """Generate Python method."""
        name = method["operation_id"]
        http_method = method["method"].lower()
        path = method["path"]
        
        code = f'''    def {name}(self, **kwargs: Any) -> dict[str, Any]:
        """
        {method.get("summary", name)}
        
        {method.get("description", "")}
        """
        response = self.client.{http_method}("{path}", **kwargs)
        response.raise_for_status()
        return response.json()
'''
        
        if include_async:
            code += f'''
    async def {name}_async(self, **kwargs: Any) -> dict[str, Any]:
        """{method.get("summary", name)} (async)."""
        response = await self.async_client.{http_method}("{path}", **kwargs)
        response.raise_for_status()
        return response.json()
'''
        return code
    
    def _python_models(self, config: SDKConfig) -> str:
        """Generate Python models."""
        schemas = self.spec.get("components", {}).get("schemas", {})
        models = []
        
        for name, schema in schemas.items():
            props = schema.get("properties", {})
            fields = ", ".join(
                f'{p}: {self._python_type(v.get("type", "Any"))} = None'
                for p, v in props.items()
            )
            models.append(f'''
@dataclass
class {name}:
    """{schema.get("description", name)}."""
    {fields if fields else "pass"}
''')
        
        return f'''"""Data models."""
from dataclasses import dataclass
from typing import Any, Optional, List

{"".join(models)}
'''
    
    def _python_type(self, openapi_type: str) -> str:
        """Convert OpenAPI type to Python type."""
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "dict[str, Any]",
        }
        return type_map.get(openapi_type, "Any")
    
    def _python_setup(self, config: SDKConfig) -> str:
        """Generate setup.py."""
        return f'''from setuptools import setup, find_packages

setup(
    name="{config.package_name}",
    version="{config.version}",
    packages=find_packages(),
    install_requires=["httpx>=0.24.0"],
    author="{config.author}",
    license="{config.license}",
    python_requires=">=3.10",
)
'''
    
    def _generate_typescript(self, config: SDKConfig) -> Path:
        """Generate TypeScript SDK."""
        pkg_dir = config.output_dir
        pkg_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate index.ts
        client_content = self._typescript_client(config)
        (pkg_dir / "index.ts").write_text(client_content)
        
        # Generate package.json
        pkg_json = self._typescript_package_json(config)
        (pkg_dir / "package.json").write_text(json.dumps(pkg_json, indent=2))
        
        return config.output_dir
    
    def _typescript_client(self, config: SDKConfig) -> str:
        """Generate TypeScript client."""
        methods = self._extract_methods()
        method_code = "\n".join(
            self._typescript_method(m) for m in methods
        )
        
        return f'''/**
 * {config.package_name} - Auto-generated SDK
 * Version: {config.version}
 */

export interface ClientConfig {{
  baseUrl?: string;
  apiKey?: string;
}}

export class RoboticsClient {{
  private baseUrl: string;
  private apiKey?: string;

  constructor(config: ClientConfig = {{}}) {{
    this.baseUrl = config.baseUrl || "{config.base_url}";
    this.apiKey = config.apiKey;
  }}

  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {{
    const headers: Record<string, string> = {{
      "Content-Type": "application/json",
    }};
    if (this.apiKey) {{
      headers["Authorization"] = `Bearer ${{this.apiKey}}`;
    }}

    const response = await fetch(`${{this.baseUrl}}${{path}}`, {{
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    }});

    if (!response.ok) {{
      throw new Error(`API error: ${{response.status}}`);
    }}

    return response.json();
  }}

{method_code}
}}

export default RoboticsClient;
'''
    
    def _typescript_method(self, method: dict) -> str:
        """Generate TypeScript method."""
        name = self._camel_case(method["operation_id"])
        http_method = method["method"].upper()
        path = method["path"]
        
        return f'''  async {name}(params?: Record<string, unknown>): Promise<unknown> {{
    return this.request("{http_method}", "{path}", params);
  }}
'''
    
    def _typescript_package_json(self, config: SDKConfig) -> dict:
        """Generate package.json."""
        return {
            "name": config.package_name,
            "version": config.version,
            "main": "dist/index.js",
            "types": "dist/index.d.ts",
            "scripts": {
                "build": "tsc",
                "test": "jest",
            },
            "devDependencies": {
                "typescript": "^5.0.0",
            },
        }
    
    def _generate_go(self, config: SDKConfig) -> Path:
        """Generate Go SDK (placeholder)."""
        config.output_dir.mkdir(parents=True, exist_ok=True)
        # Go generation would go here
        return config.output_dir
    
    def _extract_methods(self) -> list[dict]:
        """Extract API methods from spec."""
        methods = []
        paths = self.spec.get("paths", {})
        
        for path, path_item in paths.items():
            for method in ["get", "post", "put", "delete", "patch"]:
                if method in path_item:
                    op = path_item[method]
                    methods.append({
                        "path": path,
                        "method": method,
                        "operation_id": op.get("operationId", f"{method}_{path}").replace("/", "_"),
                        "summary": op.get("summary", ""),
                        "description": op.get("description", ""),
                    })
        
        return methods
    
    def _camel_case(self, s: str) -> str:
        """Convert to camelCase."""
        parts = s.replace("-", "_").split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Convenience functions
def generate_python_sdk(
    openapi_path: str,
    output_dir: str = "./python_sdk",
    package_name: str = "robotics_sdk",
) -> Path:
    """Generate Python SDK."""
    generator = SDKGenerator(openapi_path)
    config = SDKConfig(
        language=Language.PYTHON,
        output_dir=Path(output_dir),
        package_name=package_name,
    )
    return generator.generate(config)


def generate_typescript_sdk(
    openapi_path: str,
    output_dir: str = "./typescript_sdk",
    package_name: str = "robotics-sdk",
) -> Path:
    """Generate TypeScript SDK."""
    generator = SDKGenerator(openapi_path)
    config = SDKConfig(
        language=Language.TYPESCRIPT,
        output_dir=Path(output_dir),
        package_name=package_name,
    )
    return generator.generate(config)
