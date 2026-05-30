import re
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable, Iterable
from collections import Counter


CustomValidatorFunc = Callable[[Any, Any, str, "JsonSchemaValidator", List["ValidationError"]], None]


class ReferenceResolutionError(Exception):
    def __init__(self, message: str, ref: str = ""):
        super().__init__(message)
        self.ref = ref


DRAFT_URIS = {
    "http://json-schema.org/draft-04/schema#": "draft-04",
    "http://json-schema.org/draft-06/schema#": "draft-06",
    "http://json-schema.org/draft-07/schema#": "draft-07",
    "https://json-schema.org/draft/2019-09/schema": "2019-09",
    "https://json-schema.org/draft/2020-12/schema": "2020-12",
}

DRAFT_FEATURES = {
    "draft-04": {"exclusiveMaximum": "as_bool", "exclusiveMinimum": "as_bool", "if": False, "then": False, "else": False, "const": False, "contains": False, "propertyNames": False},
    "draft-06": {"exclusiveMaximum": "as_number", "exclusiveMinimum": "as_number", "if": False, "then": False, "else": False, "const": True, "contains": True, "propertyNames": True},
    "draft-07": {"exclusiveMaximum": "as_number", "exclusiveMinimum": "as_number", "if": True, "then": True, "else": True, "const": True, "contains": True, "propertyNames": True},
    "2019-09": {"exclusiveMaximum": "as_number", "exclusiveMinimum": "as_number", "if": True, "then": True, "else": True, "const": True, "contains": True, "propertyNames": True},
    "2020-12": {"exclusiveMaximum": "as_number", "exclusiveMinimum": "as_number", "if": True, "then": True, "else": True, "const": True, "contains": True, "propertyNames": True},
}


def detect_draft(schema: Dict[str, Any]) -> str:
    if "$schema" in schema:
        schema_uri = schema["$schema"]
        if schema_uri in DRAFT_URIS:
            return DRAFT_URIS[schema_uri]
        for uri, draft in DRAFT_URIS.items():
            if schema_uri.startswith(uri.replace("/schema#", "").replace("/schema", "")):
                return draft
    if "const" in schema or "contains" in schema or "propertyNames" in schema:
        return "draft-06"
    if "if" in schema or "then" in schema or "else" in schema:
        return "draft-07"
    if ("exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float))) or \
       ("exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float))):
        return "draft-06"
    return "draft-04"


@dataclass
class ValidationError:
    path: str
    message: str
    validator: str
    value: Any
    schema: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "message": self.message,
            "validator": self.validator,
            "value": repr(self.value),
            "schema": self.schema,
        }


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "error_count": len(self.errors),
        }

    def __bool__(self) -> bool:
        return self.valid


TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}

FORMAT_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "uri": r"^[a-zA-Z][a-zA-Z0-9+\-.]*://",
    "date": r"^\d{4}-\d{2}-\d{2}$",
    "time": r"^\d{2}:\d{2}:\d{2}",
    "date-time": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",
    "ipv4": r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$",
    "ipv6": r"^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
}


class ReferenceResolver:

    def __init__(self, base_uri: str = "", timeout: int = 10):
        self.base_uri = base_uri
        self.timeout = timeout
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._resolving_stack: List[str] = []

    def cache_schema(self, uri: str, schema: Dict[str, Any]) -> None:
        self._cache[uri] = schema

    def clear_cache(self) -> None:
        self._cache.clear()

    def is_remote_ref(self, ref: str) -> bool:
        parsed = urllib.parse.urlparse(ref)
        return parsed.scheme in ("http", "https", "file")

    def _parse_ref(self, ref: str) -> Tuple[str, str]:
        parsed = urllib.parse.urlparse(ref)
        if parsed.scheme or parsed.netloc:
            base_part = urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, "", "", "")
            )
            fragment = parsed.fragment
            return base_part, fragment
        if ref.startswith("#"):
            return "", ref[1:]
        if "#" in ref:
            idx = ref.find("#")
            return ref[:idx], ref[idx + 1 :]
        return ref, ""

    def _resolve_json_pointer(
        self, doc: Any, pointer: str, ref: str = ""
    ) -> Any:
        if not pointer:
            return doc
        if not pointer.startswith("/") and pointer != "":
            raise ReferenceResolutionError(
                f"Invalid JSON Pointer: {pointer}", ref
            )
        current = doc
        tokens = pointer.split("/")[1:] if pointer else []
        for i, token in enumerate(tokens):
            decoded_token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                if decoded_token not in current:
                    raise ReferenceResolutionError(
                        f"Cannot find '{decoded_token}' in schema at JSON Pointer segment {i} of '{pointer}'",
                        ref,
                    )
                current = current[decoded_token]
            elif isinstance(current, list):
                try:
                    idx = int(decoded_token)
                    if idx < 0 or idx >= len(current):
                        raise ReferenceResolutionError(
                            f"Array index {idx} out of bounds at JSON Pointer segment {i} of '{pointer}'",
                            ref,
                        )
                    current = current[idx]
                except ValueError:
                    raise ReferenceResolutionError(
                        f"Invalid array index '{decoded_token}' at JSON Pointer segment {i} of '{pointer}'",
                        ref,
                    )
            else:
                raise ReferenceResolutionError(
                    f"Cannot traverse non-dict/non-array value at JSON Pointer segment {i} of '{pointer}'",
                    ref,
                )
        return current

    def _fetch_remote(self, uri: str) -> Dict[str, Any]:
        if uri in self._cache:
            return self._cache[uri]

        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme == "file":
            filepath = urllib.parse.unquote(parsed.path)
            if os.name == "nt" and filepath.startswith("/"):
                filepath = filepath[1:]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    schema = json.load(f)
            except FileNotFoundError:
                raise ReferenceResolutionError(
                    f"Cannot find schema file: {filepath}", uri
                )
            except json.JSONDecodeError as e:
                raise ReferenceResolutionError(
                    f"Invalid JSON in file {filepath}: {e}", uri
                )
        elif parsed.scheme in ("http", "https"):
            try:
                req = urllib.request.Request(
                    uri, headers={"Accept": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    charset = resp.headers.get_content_charset() or "utf-8"
                    raw = resp.read().decode(charset)
                    schema = json.loads(raw)
            except urllib.error.URLError as e:
                raise ReferenceResolutionError(
                    f"Failed to fetch remote schema {uri}: {e}", uri
                )
            except json.JSONDecodeError as e:
                raise ReferenceResolutionError(
                    f"Invalid JSON from {uri}: {e}", uri
                )
        else:
            raise ReferenceResolutionError(
                f"Unsupported URI scheme: {parsed.scheme}", uri
            )

        self._cache[uri] = schema
        return schema

    def resolve(
        self,
        ref: str,
        current_schema: Optional[Dict[str, Any]] = None,
        current_base: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        base_part, fragment = self._parse_ref(ref)

        if not base_part or not self.is_remote_ref(ref):
            target_base = urllib.parse.urljoin(
                current_base or self.base_uri, base_part
            )
        else:
            target_base = base_part if base_part else (current_base or self.base_uri)

        resolution_key = f"{target_base}#{fragment}" if target_base else f"#{fragment}"

        if resolution_key in self._resolving_stack:
            cycle = " -> ".join(
                self._resolving_stack + [resolution_key]
            )
            raise ReferenceResolutionError(
                f"Circular reference detected: {cycle}", ref
            )

        self._resolving_stack.append(resolution_key)

        try:
            doc: Any
            if not base_part and not current_base:
                if current_schema is None:
                    raise ReferenceResolutionError(
                        "Cannot resolve local reference without base schema", ref
                    )
                root_doc = current_schema
                resolved = self._resolve_json_pointer(
                    current_schema, fragment, ref
                )
                resolved_base = current_base
            else:
                if target_base:
                    if base_part and self.is_remote_ref(base_part):
                        doc = self._fetch_remote(target_base)
                    elif target_base in self._cache:
                        doc = self._cache[target_base]
                    elif current_schema and not base_part:
                        doc = current_schema
                    else:
                        doc = self._fetch_remote(target_base)
                    resolved_base = target_base
                else:
                    doc = current_schema
                    resolved_base = current_base

                root_doc = doc
                resolved = self._resolve_json_pointer(doc, fragment, ref)

            if not isinstance(resolved, dict):
                raise ReferenceResolutionError(
                    f"Reference resolved to a non-schema value (type={type(resolved).__name__})",
                    ref,
                )

            if "$ref" in resolved:
                new_ref = resolved["$ref"]
                new_ref_base, _ = self._parse_ref(new_ref)
                new_root = root_doc if (not new_ref_base or new_ref.startswith("#")) else None
                resolved, resolved_base = self.resolve(
                    new_ref, new_root, resolved_base
                )

            return resolved, resolved_base
        finally:
            self._resolving_stack.pop()


class JsonSchemaValidator:
    _global_custom_validators: Dict[str, CustomValidatorFunc] = {}

    @classmethod
    def register_keyword(cls, keyword: str, validator_func: CustomValidatorFunc) -> None:
        cls._global_custom_validators[keyword] = validator_func

    def __init__(
        self,
        schema: Dict[str, Any],
        resolver: Optional[ReferenceResolver] = None,
        base_uri: str = "",
        draft: Optional[str] = None,
    ):
        self.schema = schema
        self._resolver = resolver or ReferenceResolver(base_uri=base_uri)
        self._base_uri = base_uri
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._custom_validators: Dict[str, CustomValidatorFunc] = {}
        self.draft = draft or detect_draft(schema)
        if base_uri and base_uri not in self._resolver._cache:
            self._resolver.cache_schema(base_uri, schema)

    def register_local_keyword(self, keyword: str, validator_func: CustomValidatorFunc) -> None:
        self._custom_validators[keyword] = validator_func

    def _get_all_validators(self) -> Dict[str, CustomValidatorFunc]:
        validators = dict(self._global_custom_validators)
        validators.update(self._custom_validators)
        return validators

    def _supports_feature(self, feature: str) -> bool:
        features = DRAFT_FEATURES.get(self.draft, DRAFT_FEATURES["draft-07"])
        val = features.get(feature, True)
        return val if isinstance(val, bool) else True

    def _feature_mode(self, feature: str) -> Any:
        features = DRAFT_FEATURES.get(self.draft, DRAFT_FEATURES["draft-07"])
        return features.get(feature, None)

    def validate(self, data: Any) -> ValidationResult:
        errors: List[ValidationError] = []
        try:
            self._validate_node(data, self.schema, "$", self._base_uri, errors)
        except ReferenceResolutionError as e:
            errors.append(
                ValidationError(
                    path="$",
                    message=f"Reference resolution error: {e}",
                    validator="$ref",
                    value=None,
                    schema=e.ref,
                )
            )
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _resolve_ref(
        self,
        schema: Dict[str, Any],
        current_base: str,
        path: str,
        errors: List[ValidationError],
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        try:
            return self._resolver.resolve(
                schema["$ref"], self.schema, current_base
            )
        except ReferenceResolutionError as e:
            errors.append(
                ValidationError(
                    path=path,
                    message=str(e),
                    validator="$ref",
                    value=None,
                    schema=e.ref,
                )
            )
            return None

    def _validate_node(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        if "$ref" in schema:
            resolved = self._resolve_ref(schema, current_base, path, errors)
            if resolved is None:
                return
            schema, current_base = resolved

        if "type" in schema:
            self._validate_type(data, schema, path, errors)
        if "enum" in schema:
            self._validate_enum(data, schema, path, errors)
        if "const" in schema and self._supports_feature("const"):
            self._validate_const(data, schema, path, errors)
        self._validate_numeric(data, schema, path, errors)
        self._validate_string(data, schema, path, errors)
        self._validate_array(data, schema, path, current_base, errors)
        self._validate_object(data, schema, path, current_base, errors)
        if "allOf" in schema:
            self._validate_all_of(data, schema, path, current_base, errors)
        if "anyOf" in schema:
            self._validate_any_of(data, schema, path, current_base, errors)
        if "oneOf" in schema:
            self._validate_one_of(data, schema, path, current_base, errors)
        if "not" in schema:
            self._validate_not(data, schema, path, current_base, errors)
        if "if" in schema and self._supports_feature("if"):
            self._validate_if_then_else(data, schema, path, current_base, errors)

        custom_validators = self._get_all_validators()
        for keyword, validator_func in custom_validators.items():
            if keyword in schema:
                validator_func(data, schema[keyword], path, self, errors)

    def _validate_type(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        expected = schema["type"]
        if isinstance(expected, list):
            matched = any(self._check_type(data, t) for t in expected)
            if not matched:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Expected type {expected}, got {type(data).__name__}",
                        validator="type",
                        value=data,
                        schema=expected,
                    )
                )
        else:
            if not self._check_type(data, expected):
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Expected type '{expected}', got '{type(data).__name__}'",
                        validator="type",
                        value=data,
                        schema=expected,
                    )
                )

    def _check_type(self, data: Any, type_name: str) -> bool:
        if type_name not in TYPE_MAP:
            return False
        expected_type = TYPE_MAP[type_name]
        if type_name == "integer":
            return isinstance(data, int) and not isinstance(data, bool)
        if type_name == "number":
            return isinstance(data, (int, float)) and not isinstance(data, bool)
        if type_name == "boolean":
            return isinstance(data, bool)
        return isinstance(data, expected_type)

    def _validate_enum(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        if data not in schema["enum"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Value must be one of {schema['enum']}",
                    validator="enum",
                    value=data,
                    schema=schema["enum"],
                )
            )

    def _validate_const(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        if data != schema["const"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Value must be {schema['const']!r}",
                    validator="const",
                    value=data,
                    schema=schema["const"],
                )
            )

    def _validate_numeric(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return

        exclusive_mode = self._feature_mode("exclusiveMaximum")

        if "minimum" in schema:
            min_val = schema["minimum"]
            exclusive_min = False
            if exclusive_mode == "as_bool":
                exclusive_min = schema.get("exclusiveMinimum", False) is True
            if exclusive_min:
                if data <= min_val:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Value {data} must be greater than {min_val}",
                            validator="exclusiveMinimum",
                            value=data,
                            schema=True,
                        )
                    )
            else:
                if data < min_val:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Value {data} is less than minimum {min_val}",
                            validator="minimum",
                            value=data,
                            schema=min_val,
                        )
                    )

        if "maximum" in schema:
            max_val = schema["maximum"]
            exclusive_max = False
            if exclusive_mode == "as_bool":
                exclusive_max = schema.get("exclusiveMaximum", False) is True
            if exclusive_max:
                if data >= max_val:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Value {data} must be less than {max_val}",
                            validator="exclusiveMaximum",
                            value=data,
                            schema=True,
                        )
                    )
            else:
                if data > max_val:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Value {data} is greater than maximum {max_val}",
                            validator="maximum",
                            value=data,
                            schema=max_val,
                        )
                    )

        if exclusive_mode == "as_number":
            if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)):
                if data <= schema["exclusiveMinimum"]:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Value {data} must be greater than {schema['exclusiveMinimum']}",
                            validator="exclusiveMinimum",
                            value=data,
                            schema=schema["exclusiveMinimum"],
                        )
                    )

            if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)):
                if data >= schema["exclusiveMaximum"]:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Value {data} must be less than {schema['exclusiveMaximum']}",
                            validator="exclusiveMaximum",
                            value=data,
                            schema=schema["exclusiveMaximum"],
                        )
                    )

        if "multipleOf" in schema:
            divisor = schema["multipleOf"]
            if divisor == 0:
                return
            quotient = data / divisor
            if abs(quotient - round(quotient)) > 1e-9:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value {data} is not a multiple of {divisor}",
                        validator="multipleOf",
                        value=data,
                        schema=divisor,
                    )
                )

    def _validate_string(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        if not isinstance(data, str):
            return

        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"String length {len(data)} is less than minLength {schema['minLength']}",
                    validator="minLength",
                    value=data,
                    schema=schema["minLength"],
                )
            )

        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"String length {len(data)} exceeds maxLength {schema['maxLength']}",
                    validator="maxLength",
                    value=data,
                    schema=schema["maxLength"],
                )
            )

        if "pattern" in schema:
            pattern = schema["pattern"]
            if pattern not in self._compiled_patterns:
                try:
                    self._compiled_patterns[pattern] = re.compile(pattern)
                except re.error as e:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Invalid regex pattern '{pattern}': {e}",
                            validator="pattern",
                            value=data,
                            schema=pattern,
                        )
                    )
                    return
            if not self._compiled_patterns[pattern].search(data):
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"String '{data}' does not match pattern '{pattern}'",
                        validator="pattern",
                        value=data,
                        schema=pattern,
                    )
                )

        if "format" in schema:
            fmt = schema["format"]
            if fmt in FORMAT_PATTERNS:
                fmt_pattern = FORMAT_PATTERNS[fmt]
                if not re.search(fmt_pattern, data):
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"String '{data}' does not match format '{fmt}'",
                            validator="format",
                            value=data,
                            schema=fmt,
                        )
                    )

    def _validate_array(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        if not isinstance(data, list):
            return

        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Array has {len(data)} items, minimum is {schema['minItems']}",
                    validator="minItems",
                    value=data,
                    schema=schema["minItems"],
                )
            )

        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Array has {len(data)} items, maximum is {schema['maxItems']}",
                    validator="maxItems",
                    value=data,
                    schema=schema["maxItems"],
                )
            )

        if "uniqueItems" in schema and schema["uniqueItems"]:
            seen = []
            for i, item in enumerate(data):
                if item in seen:
                    errors.append(
                        ValidationError(
                            path=f"{path}[{i}]",
                            message=f"Duplicate item found at index {i}",
                            validator="uniqueItems",
                            value=item,
                            schema=True,
                        )
                    )
                else:
                    seen.append(item)

        if "items" in schema:
            item_schema = schema["items"]
            if isinstance(item_schema, dict):
                for i, item in enumerate(data):
                    self._validate_node(item, item_schema, f"{path}[{i}]", current_base, errors)
            elif isinstance(item_schema, list):
                for i, item in enumerate(data):
                    if i < len(item_schema):
                        self._validate_node(
                            item, item_schema[i], f"{path}[{i}]", current_base, errors
                        )
                    elif "additionalItems" in schema:
                        additional = schema["additionalItems"]
                        if additional is False:
                            errors.append(
                                ValidationError(
                                    path=f"{path}[{i}]",
                                    message=f"Additional item at index {i} is not allowed",
                                    validator="additionalItems",
                                    value=item,
                                    schema=False,
                                )
                            )
                        elif isinstance(additional, dict):
                            self._validate_node(
                                item, additional, f"{path}[{i}]", current_base, errors
                            )

        if "contains" in schema and self._supports_feature("contains"):
            contains_schema = schema["contains"]
            found = False
            for item in data:
                temp_errors: List[ValidationError] = []
                self._validate_node(item, contains_schema, "$", current_base, temp_errors)
                if not temp_errors:
                    found = True
                    break
            if not found:
                errors.append(
                    ValidationError(
                        path=path,
                        message="Array does not contain any item matching 'contains' schema",
                        validator="contains",
                        value=data,
                        schema=contains_schema,
                    )
                )

    def _validate_object(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        if not isinstance(data, dict):
            return

        if "required" in schema:
            for field_name in schema["required"]:
                if field_name not in data:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Required field '{field_name}' is missing",
                            validator="required",
                            value=None,
                            schema=field_name,
                        )
                    )

        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in data:
                    self._validate_node(
                        data[prop_name], prop_schema, f"{path}.{prop_name}", current_base, errors
                    )

        if "patternProperties" in schema:
            for pattern, prop_schema in schema["patternProperties"].items():
                if pattern not in self._compiled_patterns:
                    try:
                        self._compiled_patterns[pattern] = re.compile(pattern)
                    except re.error:
                        continue
                compiled = self._compiled_patterns[pattern]
                for prop_name in data:
                    if compiled.search(prop_name):
                        self._validate_node(
                            data[prop_name], prop_schema, f"{path}.{prop_name}", current_base, errors
                        )

        if "additionalProperties" in schema:
            allowed_keys = set()
            if "properties" in schema:
                allowed_keys.update(schema["properties"].keys())
            if "patternProperties" in schema:
                for pattern in schema["patternProperties"]:
                    if pattern not in self._compiled_patterns:
                        try:
                            self._compiled_patterns[pattern] = re.compile(pattern)
                        except re.error:
                            continue
                    compiled = self._compiled_patterns[pattern]
                    for key in data:
                        if compiled.search(key):
                            allowed_keys.add(key)

            additional = schema["additionalProperties"]
            for key in data:
                if key not in allowed_keys:
                    if additional is False:
                        errors.append(
                            ValidationError(
                                path=f"{path}.{key}",
                                message=f"Additional property '{key}' is not allowed",
                                validator="additionalProperties",
                                value=data[key],
                                schema=False,
                            )
                        )
                    elif isinstance(additional, dict):
                        self._validate_node(
                            data[key], additional, f"{path}.{key}", current_base, errors
                        )

        if "minProperties" in schema and len(data) < schema["minProperties"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Object has {len(data)} properties, minimum is {schema['minProperties']}",
                    validator="minProperties",
                    value=data,
                    schema=schema["minProperties"],
                )
            )

        if "maxProperties" in schema and len(data) > schema["maxProperties"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Object has {len(data)} properties, maximum is {schema['maxProperties']}",
                    validator="maxProperties",
                    value=data,
                    schema=schema["maxProperties"],
                )
            )

        if "propertyNames" in schema and self._supports_feature("propertyNames"):
            name_schema = schema["propertyNames"]
            for prop_name in data:
                temp_errors: List[ValidationError] = []
                self._validate_node(prop_name, name_schema, f"{path}.{prop_name}", current_base, temp_errors)
                if temp_errors:
                    errors.append(
                        ValidationError(
                            path=f"{path}.{prop_name}",
                            message=f"Property name '{prop_name}' does not match propertyNames schema",
                            validator="propertyNames",
                            value=prop_name,
                            schema=name_schema,
                        )
                    )

        if "dependencies" in schema:
            for prop_name, dependency in schema["dependencies"].items():
                if prop_name in data:
                    if isinstance(dependency, list):
                        for dep_name in dependency:
                            if dep_name not in data:
                                errors.append(
                                    ValidationError(
                                        path=path,
                                        message=f"Property '{prop_name}' requires '{dep_name}' to be present",
                                        validator="dependencies",
                                        value=None,
                                        schema=dependency,
                                    )
                                )
                    elif isinstance(dependency, dict):
                        self._validate_node(data, dependency, path, current_base, errors)

    def _validate_all_of(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        for i, sub_schema in enumerate(schema["allOf"]):
            self._validate_node(data, sub_schema, path, current_base, errors)

    def _validate_any_of(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        for sub_schema in schema["anyOf"]:
            temp_errors: List[ValidationError] = []
            self._validate_node(data, sub_schema, path, current_base, temp_errors)
            if not temp_errors:
                return
        errors.append(
            ValidationError(
                path=path,
                message="Value does not match any of the 'anyOf' schemas",
                validator="anyOf",
                value=data,
                schema=schema["anyOf"],
            )
        )

    def _validate_one_of(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        match_count = 0
        for sub_schema in schema["oneOf"]:
            temp_errors: List[ValidationError] = []
            self._validate_node(data, sub_schema, path, current_base, temp_errors)
            if not temp_errors:
                match_count += 1
        if match_count == 0:
            errors.append(
                ValidationError(
                    path=path,
                    message="Value does not match any of the 'oneOf' schemas",
                    validator="oneOf",
                    value=data,
                    schema=schema["oneOf"],
                )
            )
        elif match_count > 1:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Value matches {match_count} schemas in 'oneOf', but must match exactly one",
                    validator="oneOf",
                    value=data,
                    schema=schema["oneOf"],
                )
            )

    def _validate_not(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        temp_errors: List[ValidationError] = []
        self._validate_node(data, schema["not"], path, current_base, temp_errors)
        if not temp_errors:
            errors.append(
                ValidationError(
                    path=path,
                    message="Value must not match the 'not' schema",
                    validator="not",
                    value=data,
                    schema=schema["not"],
                )
            )

    def _validate_if_then_else(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        current_base: str,
        errors: List[ValidationError],
    ) -> None:
        if_schema = schema["if"]
        temp_errors: List[ValidationError] = []
        self._validate_node(data, if_schema, path, current_base, temp_errors)
        condition_met = len(temp_errors) == 0

        if condition_met and "then" in schema:
            self._validate_node(data, schema["then"], path, current_base, errors)
        elif not condition_met and "else" in schema:
            self._validate_node(data, schema["else"], path, current_base, errors)


class SchemaInferer:

    def __init__(
        self,
        detect_formats: bool = True,
        detect_patterns: bool = True,
        detect_ranges: bool = True,
        infer_required: bool = True,
        merge_arrays: bool = True,
        default_draft: str = "draft-07",
    ):
        self.detect_formats = detect_formats
        self.detect_patterns = detect_patterns
        self.detect_ranges = detect_ranges
        self.infer_required = infer_required
        self.merge_arrays = merge_arrays
        self.default_draft = default_draft
        self._inferred: Optional[Dict[str, Any]] = None
        self._sample_count: int = 0

    def _get_type(self, data: Any) -> str:
        if data is None:
            return "null"
        if isinstance(data, bool):
            return "boolean"
        if isinstance(data, int):
            return "integer"
        if isinstance(data, float):
            return "number"
        if isinstance(data, str):
            return "string"
        if isinstance(data, list):
            return "array"
        if isinstance(data, dict):
            return "object"
        return "any"

    def _detect_format(self, s: str) -> Optional[str]:
        if not self.detect_formats:
            return None
        for fmt, pattern in FORMAT_PATTERNS.items():
            if re.search(pattern, s):
                return fmt
        return None

    def _infer_string_schema(self, s: str, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": "string"}
        if existing is None:
            existing = {}

        current_len = len(s)
        if self.detect_ranges:
            if existing:
                schema["minLength"] = min(existing.get("minLength", current_len), current_len)
                schema["maxLength"] = max(existing.get("maxLength", current_len), current_len)
            else:
                schema["minLength"] = current_len
                schema["maxLength"] = current_len

        detected_format = self._detect_format(s)
        if detected_format:
            if existing and existing.get("format"):
                if existing["format"] != detected_format:
                    schema.pop("format", None)
                else:
                    schema["format"] = detected_format
            else:
                schema["format"] = detected_format

        return schema

    def _infer_numeric_schema(self, n: Union[int, float], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(n, int) and not isinstance(n, bool):
            schema: Dict[str, Any] = {"type": "integer"}
        else:
            schema = {"type": "number"}

        if existing is None:
            existing = {}

        if self.detect_ranges:
            if existing:
                if existing.get("type") == "integer" and schema["type"] == "number":
                    schema["type"] = "number"
                schema["minimum"] = min(existing.get("minimum", n), n)
                schema["maximum"] = max(existing.get("maximum", n), n)
            else:
                schema["minimum"] = n
                schema["maximum"] = n

        return schema

    def _merge_schemas(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, Any]:
        if s1.get("type") != s2.get("type"):
            type1 = s1.get("type")
            type2 = s2.get("type")
            types = []
            if isinstance(type1, list):
                types.extend(type1)
            elif type1:
                types.append(type1)
            if isinstance(type2, list):
                types.extend(type2)
            elif type2:
                types.append(type2)
            unique_types = []
            for t in types:
                if t not in unique_types:
                    unique_types.append(t)
            if len(unique_types) == 1:
                result = {"type": unique_types[0]}
            else:
                result = {"type": unique_types}
            return result

        result = dict(s1)
        schema_type = s1.get("type")

        if schema_type == "object":
            props1 = s1.get("properties", {})
            props2 = s2.get("properties", {})
            all_props = {}
            for key in set(props1.keys()) | set(props2.keys()):
                if key in props1 and key in props2:
                    all_props[key] = self._merge_schemas(props1[key], props2[key])
                elif key in props1:
                    all_props[key] = props1[key]
                else:
                    all_props[key] = props2[key]
            if all_props:
                result["properties"] = all_props

            if self.infer_required:
                req1 = set(s1.get("required", []))
                req2 = set(s2.get("required", []))
                common_required = list(req1 & req2)
                if common_required:
                    result["required"] = sorted(common_required)
                elif "required" in result:
                    del result["required"]

        elif schema_type == "array":
            items1 = s1.get("items")
            items2 = s2.get("items")
            if items1 and items2:
                result["items"] = self._merge_schemas(items1, items2)
            elif items2:
                result["items"] = items2

            if self.detect_ranges:
                items1_len = s1.get("minItems")
                items2_len = s2.get("minItems")
                if items1_len is not None and items2_len is not None:
                    result["minItems"] = min(items1_len, items2_len)

        elif schema_type == "string":
            if self.detect_ranges:
                result["minLength"] = min(s1.get("minLength", 0), s2.get("minLength", 0))
                result["maxLength"] = max(s1.get("maxLength", 0), s2.get("maxLength", 0))

            fmt1 = s1.get("format")
            fmt2 = s2.get("format")
            if fmt1 and fmt2 and fmt1 == fmt2:
                result["format"] = fmt1
            elif "format" in result:
                del result["format"]

        elif schema_type in ("integer", "number"):
            if s1.get("type") == "integer" and s2.get("type") == "number":
                result["type"] = "number"
            if self.detect_ranges:
                result["minimum"] = min(s1.get("minimum", 0), s2.get("minimum", 0))
                result["maximum"] = max(s1.get("maximum", 0), s2.get("maximum", 0))

        return result

    def _infer_schema(self, data: Any, existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data_type = self._get_type(data)

        if data_type == "null":
            return {"type": "null"}

        if data_type == "boolean":
            return {"type": "boolean"}

        if data_type in ("integer", "number"):
            return self._infer_numeric_schema(data, existing)

        if data_type == "string":
            return self._infer_string_schema(data, existing)

        if data_type == "array":
            schema: Dict[str, Any] = {"type": "array"}
            if data:
                item_schema = None
                if self.merge_arrays:
                    for item in data:
                        item_s = self._infer_schema(item, item_schema)
                        if item_schema is None:
                            item_schema = item_s
                        else:
                            item_schema = self._merge_schemas(item_schema, item_s)
                    if item_schema:
                        schema["items"] = item_schema
                else:
                    schema["items"] = {"type": "any"}

                if self.detect_ranges:
                    schema["minItems"] = len(data)
                    schema["maxItems"] = len(data)

            return schema

        if data_type == "object":
            schema = {"type": "object"}
            if data:
                properties = {}
                required = []
                existing_props = existing.get("properties", {}) if existing else {}
                for key, value in data.items():
                    if key in existing_props:
                        sub_schema = self._infer_schema(value, existing_props[key])
                        properties[key] = self._merge_schemas(existing_props[key], sub_schema)
                    else:
                        properties[key] = self._infer_schema(value)
                    required.append(key)
                schema["properties"] = properties
                if self.infer_required and required:
                    schema["required"] = sorted(required)
            return schema

        return {}

    def add_sample(self, data: Any) -> "SchemaInferer":
        if self._inferred is None:
            self._inferred = self._infer_schema(data)
        else:
            new_schema = self._infer_schema(data, self._inferred)
            self._inferred = self._merge_schemas(self._inferred, new_schema)
        self._sample_count += 1
        return self

    def add_samples(self, samples: Iterable[Any]) -> "SchemaInferer":
        for sample in samples:
            self.add_sample(sample)
        return self

    def to_schema(self, include_schema_uri: bool = True) -> Dict[str, Any]:
        if self._inferred is None:
            return {}
        schema = dict(self._inferred)
        if include_schema_uri:
            draft_uri = [k for k, v in DRAFT_URIS.items() if v == self.default_draft]
            if draft_uri:
                schema["$schema"] = draft_uri[0]
        schema["$inferred"] = {
            "sample_count": self._sample_count,
            "detect_formats": self.detect_formats,
            "detect_ranges": self.detect_ranges,
            "infer_required": self.infer_required,
        }
        return schema

    def reset(self) -> "SchemaInferer":
        self._inferred = None
        self._sample_count = 0
        return self


def infer_schema(
    data: Any,
    detect_formats: bool = True,
    detect_patterns: bool = True,
    detect_ranges: bool = True,
    infer_required: bool = True,
    include_schema_uri: bool = True,
) -> Dict[str, Any]:
    inferer = SchemaInferer(
        detect_formats=detect_formats,
        detect_patterns=detect_patterns,
        detect_ranges=detect_ranges,
        infer_required=infer_required,
    )
    inferer.add_sample(data)
    return inferer.to_schema(include_schema_uri=include_schema_uri)


def infer_schema_from_samples(
    samples: Iterable[Any],
    detect_formats: bool = True,
    detect_patterns: bool = True,
    detect_ranges: bool = True,
    infer_required: bool = True,
    include_schema_uri: bool = True,
) -> Dict[str, Any]:
    inferer = SchemaInferer(
        detect_formats=detect_formats,
        detect_patterns=detect_patterns,
        detect_ranges=detect_ranges,
        infer_required=infer_required,
    )
    inferer.add_samples(samples)
    return inferer.to_schema(include_schema_uri=include_schema_uri)


def validate(data: Any, schema: Dict[str, Any]) -> ValidationResult:
    validator = JsonSchemaValidator(schema)
    return validator.validate(data)


def validate_json(data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
    result = validate(data, schema)
    return result.to_dict()
